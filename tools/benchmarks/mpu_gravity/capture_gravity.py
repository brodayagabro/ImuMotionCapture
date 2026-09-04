#!/usr/bin/env python3
"""Interactive six-face MPU6050 gravity and DMP benchmark receiver."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import socket
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


UDP_PORT = 4210
ACCEL_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 16.4

FACE_ORDER = ("+Z", "-Z", "+X", "-X", "+Y", "-Y")
FACE_DESCRIPTION = {
    "+Z": "ось +Z датчика направлена строго вверх",
    "-Z": "ось -Z датчика направлена строго вверх (датчик перевёрнут)",
    "+X": "ось +X датчика направлена строго вверх",
    "-X": "ось -X датчика направлена строго вверх",
    "+Y": "ось +Y датчика направлена строго вверх",
    "-Y": "ось -Y датчика направлена строго вверх",
}
EXPECTED_ACCEL = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}

CSV_FIELDS = (
    "host_time_s",
    "device_ms",
    "sequence",
    "face",
    "sensor_id",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "qw",
    "qx",
    "qy",
    "qz",
)


class BenchmarkError(RuntimeError):
    """An expected benchmark or protocol failure."""


@dataclass(frozen=True)
class SensorRecord:
    host_time_s: float
    device_ms: int
    sequence: int
    face: str
    sensor_id: int
    ax: int
    ay: int
    az: int
    gx: int
    gy: int
    gz: int
    qw: float
    qx: float
    qy: float
    qz: float


def parse_frame(payload: bytes, host_time_s: float, face: str) -> list[SensorRecord]:
    """Parse one GRAVITY_FRAME datagram.

    Non-frame replies are ignored. Malformed benchmark frames are rejected so
    truncated UDP packets cannot silently contaminate the measurements.
    """

    text = payload.decode("ascii", errors="replace").strip()
    if not text.startswith("GRAVITY_FRAME "):
        return []

    lines = text.splitlines()
    header = lines[0].split()
    if len(header) != 4:
        raise BenchmarkError(f"Некорректный заголовок пакета: {lines[0]!r}")
    try:
        sequence = int(header[1])
        device_ms = int(header[2])
        expected_count = int(header[3])
    except ValueError as exc:
        raise BenchmarkError(f"Некорректные числа в заголовке: {lines[0]!r}") from exc

    records: list[SensorRecord] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 12 or parts[0] != "S":
            raise BenchmarkError(f"Некорректная строка датчика: {line!r}")
        try:
            records.append(
                SensorRecord(
                    host_time_s=host_time_s,
                    device_ms=device_ms,
                    sequence=sequence,
                    face=face,
                    sensor_id=int(parts[1]),
                    ax=int(parts[2]),
                    ay=int(parts[3]),
                    az=int(parts[4]),
                    gx=int(parts[5]),
                    gy=int(parts[6]),
                    gz=int(parts[7]),
                    qw=float(parts[8]),
                    qx=float(parts[9]),
                    qy=float(parts[10]),
                    qz=float(parts[11]),
                )
            )
        except ValueError as exc:
            raise BenchmarkError(f"Некорректные данные датчика: {line!r}") from exc

    if len(records) != expected_count:
        raise BenchmarkError(
            f"Пакет обещает {expected_count} датчиков, получено {len(records)}"
        )
    return records


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalized(vector: Sequence[float]) -> tuple[float, ...]:
    norm = vector_norm(vector)
    if norm < 1e-12:
        raise BenchmarkError("Получен вектор нулевой длины")
    return tuple(value / norm for value in vector)


def vector_angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    unit_first = normalized(first)
    unit_second = normalized(second)
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(unit_first, unit_second))))
    return math.degrees(math.acos(dot))


def quaternion_normalized(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    result = normalized(quaternion)
    return result[0], result[1], result[2], result[3]


def quaternion_gravity(quaternion: Sequence[float]) -> tuple[float, float, float]:
    """Return the DMP gravity direction using I2Cdevlib's convention."""

    qw, qx, qy, qz = quaternion_normalized(quaternion)
    return (
        2.0 * (qx * qz - qw * qy),
        2.0 * (qw * qx + qy * qz),
        qw * qw - qx * qx - qy * qy + qz * qz,
    )


def average_quaternion(
    quaternions: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    if not quaternions:
        raise BenchmarkError("Нельзя усреднить пустой список кватернионов")
    reference = quaternion_normalized(quaternions[0])
    accumulated = [0.0, 0.0, 0.0, 0.0]
    for value in quaternions:
        quaternion = quaternion_normalized(value)
        if sum(a * b for a, b in zip(reference, quaternion)) < 0.0:
            quaternion = tuple(-component for component in quaternion)
        for index, component in enumerate(quaternion):
            accumulated[index] += component
    return quaternion_normalized(accumulated)


def quaternion_angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    q_first = quaternion_normalized(first)
    q_second = quaternion_normalized(second)
    dot = abs(sum(a * b for a, b in zip(q_first, q_second)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def summarize_face(records: Sequence[SensorRecord]) -> dict[str, object]:
    if not records:
        raise BenchmarkError("Нет измерений для анализа")
    face = records[0].face
    sensor_id = records[0].sensor_id
    if any(record.face != face or record.sensor_id != sensor_id for record in records):
        raise BenchmarkError("В одну группу анализа попали разные датчики или грани")

    accel_vectors = [(record.ax, record.ay, record.az) for record in records]
    mean_accel_counts = tuple(
        statistics.fmean(vector[axis] for vector in accel_vectors) for axis in range(3)
    )
    accel_norms_g = [vector_norm(vector) / ACCEL_LSB_PER_G for vector in accel_vectors]
    accel_norm_mean_g = statistics.fmean(accel_norms_g)
    accel_norm_std_g = statistics.pstdev(accel_norms_g)
    face_error_deg = vector_angle_deg(mean_accel_counts, EXPECTED_ACCEL[face])

    ordered = sorted(records, key=lambda record: record.host_time_s)
    window_size = max(1, len(ordered) // 5)
    first_accel = tuple(
        statistics.fmean(
            (record.ax, record.ay, record.az)[axis]
            for record in ordered[:window_size]
        )
        for axis in range(3)
    )
    last_accel = tuple(
        statistics.fmean(
            (record.ax, record.ay, record.az)[axis]
            for record in ordered[-window_size:]
        )
        for axis in range(3)
    )
    accel_direction_drift_deg = vector_angle_deg(first_accel, last_accel)

    mismatch_angles = []
    for record in records:
        raw_accel = (record.ax, record.ay, record.az)
        gravity = quaternion_gravity((record.qw, record.qx, record.qy, record.qz))
        mismatch_angles.append(vector_angle_deg(raw_accel, gravity))
    dmp_accel_mismatch_deg = statistics.fmean(mismatch_angles)

    first_q = average_quaternion(
        [(item.qw, item.qx, item.qy, item.qz) for item in ordered[:window_size]]
    )
    last_q = average_quaternion(
        [(item.qw, item.qx, item.qy, item.qz) for item in ordered[-window_size:]]
    )
    quaternion_drift_deg = quaternion_angle_deg(first_q, last_q)
    dmp_gravity_drift_deg = vector_angle_deg(
        quaternion_gravity(first_q),
        quaternion_gravity(last_q),
    )

    gyro_rms_dps = math.sqrt(
        statistics.fmean(
            record.gx * record.gx + record.gy * record.gy + record.gz * record.gz
            for record in records
        )
    ) / GYRO_LSB_PER_DPS

    checks = {
        "sample_count": len(records) >= 10,
        "face_error": face_error_deg <= 5.0,
        "accel_norm": abs(accel_norm_mean_g - 1.0) <= 0.08,
        "accel_noise": accel_norm_std_g <= 0.02,
        "accel_direction_drift": accel_direction_drift_deg <= 2.0,
        "dmp_accel_match": dmp_accel_mismatch_deg <= 8.0,
        "dmp_gravity_drift": dmp_gravity_drift_deg <= 3.0,
        "gyro_stationary": gyro_rms_dps <= 2.0,
    }
    return {
        "sensor_id": sensor_id,
        "face": face,
        "samples": len(records),
        "mean_accel_counts": {
            "x": mean_accel_counts[0],
            "y": mean_accel_counts[1],
            "z": mean_accel_counts[2],
        },
        "accel_norm_mean_g": accel_norm_mean_g,
        "accel_norm_std_g": accel_norm_std_g,
        "face_error_deg": face_error_deg,
        "accel_direction_drift_deg": accel_direction_drift_deg,
        "dmp_accel_mismatch_deg": dmp_accel_mismatch_deg,
        "dmp_gravity_drift_deg": dmp_gravity_drift_deg,
        "quaternion_drift_deg": quaternion_drift_deg,
        "gyro_rms_dps": gyro_rms_dps,
        "checks": checks,
        "passed": all(checks.values()),
    }


def calculate_axis_calibration(face_summaries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Estimate zero bias and sensitivity from opposite stationary faces."""

    by_sensor: dict[int, dict[str, dict[str, object]]] = {}
    for summary in face_summaries:
        by_sensor.setdefault(int(summary["sensor_id"]), {})[str(summary["face"])] = summary

    result = []
    for sensor_id, faces in sorted(by_sensor.items()):
        if not all(face in faces for face in FACE_ORDER):
            continue
        axes: dict[str, dict[str, float]] = {}
        for axis_name, positive_face, negative_face in (
            ("x", "+X", "-X"),
            ("y", "+Y", "-Y"),
            ("z", "+Z", "-Z"),
        ):
            positive = float(faces[positive_face]["mean_accel_counts"][axis_name])
            negative = float(faces[negative_face]["mean_accel_counts"][axis_name])
            bias = (positive + negative) / 2.0
            scale = (positive - negative) / 2.0
            axes[axis_name] = {
                "bias_counts": bias,
                "scale_counts_per_g": scale,
                "scale_error_percent": 100.0 * (scale / ACCEL_LSB_PER_G - 1.0),
            }
        result.append({"sensor_id": sensor_id, "axes": axes})
    return result


def analyze(records: Sequence[SensorRecord]) -> dict[str, object]:
    grouped: dict[tuple[int, str], list[SensorRecord]] = {}
    for record in records:
        grouped.setdefault((record.sensor_id, record.face), []).append(record)
    face_summaries = [
        summarize_face(group)
        for _, group in sorted(
            grouped.items(),
            key=lambda item: (item[0][0], FACE_ORDER.index(item[0][1])),
        )
    ]
    sensor_ids = sorted({record.sensor_id for record in records})
    complete = all(
        (sensor_id, face) in grouped for sensor_id in sensor_ids for face in FACE_ORDER
    )
    return {
        "protocol": "GRAVITY_FRAME/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "constants": {
            "accel_lsb_per_g": ACCEL_LSB_PER_G,
            "gyro_lsb_per_dps": GYRO_LSB_PER_DPS,
        },
        "thresholds": {
            "face_error_deg_max": 5.0,
            "accel_norm_error_g_max": 0.08,
            "accel_norm_std_g_max": 0.02,
            "accel_direction_drift_deg_max": 2.0,
            "dmp_accel_mismatch_deg_max": 8.0,
            "dmp_gravity_drift_deg_max": 3.0,
            "gyro_rms_dps_max": 2.0,
        },
        "complete": complete,
        "passed": complete and all(bool(item["passed"]) for item in face_summaries),
        "face_summaries": face_summaries,
        "axis_calibration": calculate_axis_calibration(face_summaries),
    }


def write_csv(path: Path, records: Iterable[SensorRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    os.replace(temporary, path)


def write_json(path: Path, summary: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_report(summary: dict[str, object]) -> str:
    face_summaries = list(summary["face_summaries"])
    lines = [
        "# Диагностика MPU6050: ускорение и DMP",
        "",
        f"Итог: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        "| Датчик | Грань | N | Ошибка грани | Норма, g | Шум, g | Дрейф accel | DMP↔accel | Дрейф g DMP | Гиро RMS |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in face_summaries:
        marker = "✓" if item["passed"] else "✗"
        lines.append(
            "| {sensor_id} | {face} {marker} | {samples} | {face_error_deg:.2f}° | "
            "{accel_norm_mean_g:.4f} | {accel_norm_std_g:.4f} | "
            "{accel_direction_drift_deg:.2f}° | {dmp_accel_mismatch_deg:.2f}° | "
            "{dmp_gravity_drift_deg:.2f}° | "
            "{gyro_rms_dps:.2f}°/s |".format(marker=marker, **item)
        )

    lines.extend(["", "## Оценка акселерометра", ""])
    calibrations = list(summary["axis_calibration"])
    if not calibrations:
        lines.append("Нет полного набора из шести граней.")
    for calibration in calibrations:
        lines.extend(
            [
                f"Датчик {calibration['sensor_id']}:",
                "",
                "| Ось | Смещение, counts | Чувствительность, counts/g | Ошибка масштаба |",
                "|:---:|---:|---:|---:|",
            ]
        )
        for axis_name in ("x", "y", "z"):
            axis = calibration["axes"][axis_name]
            lines.append(
                f"| {axis_name.upper()} | {axis['bias_counts']:.1f} | "
                f"{axis['scale_counts_per_g']:.1f} | {axis['scale_error_percent']:+.2f}% |"
            )
        lines.append("")

    raw_failed = [
        item
        for item in face_summaries
        if not item["checks"]["face_error"]
        or not item["checks"]["accel_norm"]
        or not item["checks"]["accel_noise"]
        or not item["checks"]["accel_direction_drift"]
    ]
    dmp_failed = [
        item
        for item in face_summaries
        if not item["checks"]["dmp_accel_match"]
        or not item["checks"]["dmp_gravity_drift"]
    ]
    lines.extend(["## Интерпретация", ""])
    if raw_failed:
        lines.append(
            "- Сырые данные акселерометра не проходят часть проверок: сначала проверяйте "
            "смещения/масштаб MPU6050, питание, монтаж и качество конкретного модуля."
        )
    else:
        lines.append("- Направление и модуль сырого ускорения проходят заданные пороги.")
    if dmp_failed:
        lines.append(
            "- DMP расходится с сырым направлением тяжести или продолжает двигаться в покое: "
            "это указывает на DMP/гироскоп/прошивку, а не на визуализатор."
        )
    else:
        lines.append("- DMP согласуется с сырым ускорением и стабилен в пределах порогов.")
    lines.extend(
        [
            "",
            "Порог FAIL — диагностический, а не готовая к записи калибровка. "
            "CSV и JSON сохраняют исходные значения для повторного анализа.",
            "",
        ]
    )
    return "\n".join(lines)


def drain_socket(udp_socket: socket.socket) -> None:
    previous_timeout = udp_socket.gettimeout()
    udp_socket.setblocking(False)
    try:
        while True:
            try:
                udp_socket.recvfrom(65535)
            except BlockingIOError:
                break
    finally:
        udp_socket.settimeout(previous_timeout)


def command(
    udp_socket: socket.socket,
    target: tuple[str, int],
    text: str,
    timeout_s: float,
) -> str:
    deadline = time.monotonic() + timeout_s
    encoded = (text.strip() + "\n").encode("ascii")
    next_send = 0.0
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0.0:
            raise BenchmarkError(f"Нет ответа на команду {text!r}")
        if now >= next_send:
            udp_socket.sendto(encoded, target)
            next_send = now + 0.5
        udp_socket.settimeout(min(remaining, max(0.01, next_send - now)))
        try:
            payload, address = udp_socket.recvfrom(65535)
        except socket.timeout:
            continue
        if address[0] != target[0]:
            continue
        reply = payload.decode("ascii", errors="replace").strip()
        if reply.startswith(("ACK ", "PONG", "STATUS ", "ERR ")):
            if reply.startswith("ERR "):
                raise BenchmarkError(f"Контроллер ответил: {reply}")
            return reply


def collect_face(
    udp_socket: socket.socket,
    target: tuple[str, int],
    face: str,
    duration_s: float,
    packet_timeout_s: float,
) -> list[SensorRecord]:
    records: list[SensorRecord] = []
    deadline = time.monotonic() + duration_s
    next_progress = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        udp_socket.settimeout(min(packet_timeout_s, deadline - now))
        try:
            payload, address = udp_socket.recvfrom(65535)
        except socket.timeout as exc:
            if time.monotonic() >= deadline:
                break
            raise BenchmarkError(
                f"Поток данных пропал более чем на {packet_timeout_s:.1f} с"
            ) from exc
        if address[0] != target[0]:
            continue
        records.extend(parse_frame(payload, time.time(), face))
        if time.monotonic() >= next_progress:
            remaining = max(0.0, deadline - time.monotonic())
            print(
                f"\r  осталось {remaining:4.1f} с, строк датчиков: {len(records):5d}",
                end="",
                flush=True,
            )
            next_progress = time.monotonic() + 0.25
    print()
    if not records:
        raise BenchmarkError(f"Для грани {face} не получено ни одного измерения")
    return records


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("значение не может быть отрицательным")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Принимает сырые accel/gyro и DMP-кватернионы стендовой прошивки, "
            "проводит шестипозиционный тест MPU6050."
        )
    )
    parser.add_argument("controller_ip", help="IP-адрес ESP32 со стендовой прошивкой")
    parser.add_argument("--port", type=int, default=UDP_PORT, help=f"UDP-порт (по умолчанию {UDP_PORT})")
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=12.0,
        help="длительность записи каждой грани, с (по умолчанию 12)",
    )
    parser.add_argument(
        "--settle",
        type=nonnegative_float,
        default=1.0,
        help="пауза после Enter перед записью, с (по умолчанию 1)",
    )
    parser.add_argument(
        "--command-timeout",
        type=positive_float,
        default=3.0,
        help="таймаут ответа на команду, с",
    )
    parser.add_argument(
        "--packet-timeout",
        type=positive_float,
        default=2.0,
        help="допустимый перерыв потока, с",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="каталог результата; по умолчанию results/<дата-время>",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent
            / "results"
            / time.strftime("%Y%m%d-%H%M%S")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_samples.csv"

    try:
        controller_ip = socket.gethostbyname(args.controller_ip)
    except socket.gaierror as exc:
        raise BenchmarkError(f"Не удалось определить адрес {args.controller_ip!r}") from exc
    target = (controller_ip, args.port)
    all_records: list[SensorRecord] = []

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(("", 0))
    started = False
    try:
        hello = command(udp_socket, target, "HELLO", args.command_timeout)
        if "gravity_benchmark=1" not in hello:
            raise BenchmarkError(
                "На ESP32 запущена не стендовая прошивка gravity_benchmark"
            )
        ready_match = re.search(r"\bready=(\d+)\b", hello)
        if ready_match is not None and int(ready_match.group(1)) == 0:
            raise BenchmarkError("Прошивка не смогла инициализировать ни одного DMP")
        print(f"ESP32 {controller_ip}:{args.port}: {hello}")
        print("Во время записи не касайтесь стола и проводов с датчиками.")
        command(udp_socket, target, "START", args.command_timeout)
        started = True

        for face in FACE_ORDER:
            input(f"\nУложите ВСЕ датчики: {FACE_DESCRIPTION[face]}. Нажмите Enter...")
            drain_socket(udp_socket)
            if args.settle:
                print(f"  стабилизация {args.settle:.1f} с...")
                time.sleep(args.settle)
                drain_socket(udp_socket)
            face_records = collect_face(
                udp_socket,
                target,
                face,
                args.duration,
                args.packet_timeout,
            )
            all_records.extend(face_records)
            write_csv(raw_path, all_records)
            counts: dict[int, int] = {}
            for record in face_records:
                counts[record.sensor_id] = counts.get(record.sensor_id, 0) + 1
            print(
                "  записано: "
                + ", ".join(
                    f"датчик {sensor_id}: {count}" for sensor_id, count in sorted(counts.items())
                )
            )
    finally:
        if started:
            try:
                drain_socket(udp_socket)
                command(udp_socket, target, "STOP", args.command_timeout)
            except BenchmarkError as exc:
                print(f"Предупреждение при остановке потока: {exc}", file=sys.stderr)
        udp_socket.close()

    summary = analyze(all_records)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(f"\nРезультат: {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"Отчёт: {output_dir / 'report.md'}")
    print(f"Данные: {raw_path}")
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nТест прерван пользователем.", file=sys.stderr)
        return 130
    except (BenchmarkError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
