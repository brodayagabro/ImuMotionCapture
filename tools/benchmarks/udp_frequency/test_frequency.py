#!/usr/bin/env python3
"""Measure ESP32 UDP frame arrival times at a range of requested rates."""

from __future__ import annotations

import argparse
import csv
import socket
import sys
import time
from pathlib import Path
from typing import Final


DEFAULT_PORT: Final = 4210
DEFAULT_PACKET_COUNT: Final = 500
FREQUENCIES_HZ: Final = tuple(range(10, 151, 5))
RECEIVE_BUFFER_SIZE: Final = 4096
COMMAND_ATTEMPTS: Final = 3


class ControllerError(RuntimeError):
    """Raised when the controller rejects a command or stops responding."""


def parse_frame_sequence(payload: bytes) -> int | None:
    """Return the sequence number for a valid FRAME datagram."""
    try:
        first_line = payload.split(b"\n", 1)[0].decode("ascii")
        fields = first_line.split()
        if len(fields) != 4 or fields[0] != "FRAME":
            return None
        return int(fields[1])
    except (UnicodeDecodeError, ValueError):
        return None


def receive_until_reply(
    sock: socket.socket,
    command: str,
    expected_prefix: str,
    timeout_s: float,
) -> str:
    """Send a command and wait for its reply, ignoring streamed FRAME packets."""
    command_bytes = command.encode("ascii")
    deadline = time.monotonic() + timeout_s

    for attempt in range(1, COMMAND_ATTEMPTS + 1):
        sock.send(command_bytes)

        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                break
            sock.settimeout(remaining_s)
            try:
                payload = sock.recv(RECEIVE_BUFFER_SIZE)
            except socket.timeout:
                break

            received_at_s = time.perf_counter()
            if parse_frame_sequence(payload) is not None:
                # The timestamp is deliberately sampled immediately after recv().
                # Frames received while waiting for an ACK belong to a transition.
                _ = received_at_s
                continue

            reply = payload.decode("ascii", errors="replace").strip()
            if reply.startswith("ERR"):
                raise ControllerError(f"контроллер отклонил {command!r}: {reply}")
            if reply.startswith(expected_prefix):
                return reply

        if attempt < COMMAND_ATTEMPTS:
            deadline = time.monotonic() + timeout_s

    raise ControllerError(
        f"нет ответа {expected_prefix!r} на команду {command!r} "
        f"после {COMMAND_ATTEMPTS} попыток"
    )


def collect_frame_timestamps(
    sock: socket.socket,
    frequency_hz: int,
    packet_count: int,
    packet_timeout_s: float,
) -> list[float]:
    """Collect host monotonic timestamps for FRAME datagrams."""
    timestamps_s: list[float] = []
    previous_sequence: int | None = None
    missing_packets = 0

    while len(timestamps_s) < packet_count:
        sock.settimeout(packet_timeout_s)
        try:
            payload = sock.recv(RECEIVE_BUFFER_SIZE)
            received_at_s = time.perf_counter()
        except socket.timeout as error:
            raise ControllerError(
                f"тайм-аут {packet_timeout_s:g} с при ожидании пакета "
                f"{len(timestamps_s) + 1}/{packet_count} на {frequency_hz} Гц"
            ) from error

        sequence = parse_frame_sequence(payload)
        if sequence is None:
            reply = payload.decode("ascii", errors="replace").strip()
            if reply.startswith("ERR"):
                raise ControllerError(f"ошибка контроллера: {reply}")
            continue

        timestamps_s.append(received_at_s)
        if previous_sequence is not None and sequence > previous_sequence + 1:
            missing_packets += sequence - previous_sequence - 1
        previous_sequence = sequence

        received = len(timestamps_s)
        if received % 50 == 0 or received == packet_count:
            print(
                f"  {received:>{len(str(packet_count))}}/{packet_count} пакетов",
                end="\r" if received < packet_count else "\n",
                flush=True,
            )

    if missing_packets:
        print(f"  предупреждение: по sequence пропущено не менее {missing_packets} кадров")
    return timestamps_s


def save_table(
    output_path: Path,
    results: dict[int, list[float]],
    packet_count: int,
) -> None:
    """Atomically save one frequency per column and one timestamp per row."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")

    with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(FREQUENCIES_HZ)
        for row_index in range(packet_count):
            writer.writerow(
                f"{timestamps[row_index]:.9f}" if row_index < len(timestamps) else ""
                for frequency_hz in FREQUENCIES_HZ
                for timestamps in (results.get(frequency_hz, []),)
            )

    temporary_path.replace(output_path)


def run_test(args: argparse.Namespace) -> Path:
    endpoint = (args.ip, args.port)
    output_path = args.output.expanduser().resolve()
    results: dict[int, list[float]] = {}

    try:
        resolved_endpoint = socket.getaddrinfo(
            endpoint[0], endpoint[1], socket.AF_INET, socket.SOCK_DGRAM
        )[0][4]
    except OSError as error:
        raise ControllerError(f"не удалось определить адрес {args.ip!r}: {error}") from error

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind((args.bind_ip, args.bind_port))
        sock.connect(resolved_endpoint)
        local_ip, local_port = sock.getsockname()
        print(
            f"Контроллер: {resolved_endpoint[0]}:{resolved_endpoint[1]}; "
            f"локальный сокет: {local_ip}:{local_port}"
        )

        receive_until_reply(sock, "HELLO", "ACK HELLO", args.command_timeout)
        try:
            for frequency_hz in FREQUENCIES_HZ:
                print(f"{frequency_hz} Гц:")
                receive_until_reply(sock, "STOP", "ACK STOP", args.command_timeout)
                reply = receive_until_reply(
                    sock,
                    f"SET_RATE {frequency_hz}",
                    "ACK SET_RATE",
                    args.command_timeout,
                )
                if f"rate_hz={frequency_hz}" not in reply.split():
                    raise ControllerError(
                        f"контроллер подтвердил не ту частоту для {frequency_hz} Гц: {reply}"
                    )
                receive_until_reply(sock, "START", "ACK START", args.command_timeout)

                results[frequency_hz] = collect_frame_timestamps(
                    sock,
                    frequency_hz,
                    args.packet_count,
                    args.packet_timeout,
                )
                save_table(output_path, results, args.packet_count)
        finally:
            # Keep already measured columns even after Ctrl+C, timeout or rejection.
            save_table(output_path, results, args.packet_count)
            try:
                receive_until_reply(sock, "STOP", "ACK STOP", args.command_timeout)
            except (ControllerError, OSError):
                pass

    return output_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Запросить у ESP32 частоты 10..150 Гц с шагом 5 Гц, принять "
            "по 500 UDP FRAME-пакетов и сохранить времена приёма в CSV."
        )
    )
    parser.add_argument("ip", help="IP-адрес ESP32")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP-порт (4210)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("frequency_timestamps.csv"),
        help="выходной CSV (по умолчанию рядом со скриптом)",
    )
    parser.add_argument(
        "--packet-count",
        type=positive_int,
        default=DEFAULT_PACKET_COUNT,
        help="число FRAME-пакетов на частоту (500)",
    )
    parser.add_argument(
        "--command-timeout",
        type=positive_float,
        default=2.0,
        help="тайм-аут ответа на команду, с (2)",
    )
    parser.add_argument(
        "--packet-timeout",
        type=positive_float,
        default=3.0,
        help="тайм-аут между FRAME-пакетами, с (3)",
    )
    parser.add_argument("--bind-ip", default="0.0.0.0", help=argparse.SUPPRESS)
    parser.add_argument("--bind-port", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port должен быть в диапазоне 1..65535")
    if not 0 <= args.bind_port <= 65535:
        parser.error("--bind-port должен быть в диапазоне 0..65535")
    return args


def main() -> int:
    args = parse_args()
    try:
        output_path = run_test(args)
    except KeyboardInterrupt:
        print("\nТест прерван пользователем; накопленные данные сохранены.", file=sys.stderr)
        return 130
    except (ControllerError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    print(f"Готово: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
