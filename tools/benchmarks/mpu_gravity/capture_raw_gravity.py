#!/usr/bin/env python3
"""Two-face raw-register gravity test for MPU-6500 benchmark firmware."""

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
from typing import Sequence

from capture_gravity import (
    BenchmarkError,
    command,
    drain_socket,
    nonnegative_float,
    positive_float,
    vector_angle_deg,
    vector_norm,
)


ACCEL_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
FACES = ("+Z", "-Z")
EXPECTED = {"+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0)}
DESCRIPTIONS = {
    "+Z": "ось +Z направлена строго вверх",
    "-Z": "ось -Z направлена строго вверх (плата перевёрнута)",
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
    "temperature_raw",
    "gx",
    "gy",
    "gz",
)


@dataclass(frozen=True)
class RawRecord:
    host_time_s: float
    device_ms: int
    sequence: int
    face: str
    sensor_id: int
    ax: int
    ay: int
    az: int
    temperature_raw: int
    gx: int
    gy: int
    gz: int


def parse_raw_frame(payload: bytes, host_time_s: float, face: str) -> list[RawRecord]:
    text = payload.decode("ascii", errors="replace").strip()
    if not text.startswith("RAW_FRAME "):
        return []
    lines = text.splitlines()
    header = lines[0].split()
    if len(header) != 4:
        raise BenchmarkError(f"Некорректный RAW_FRAME: {lines[0]!r}")
    try:
        sequence = int(header[1])
        device_ms = int(header[2])
        expected_count = int(header[3])
    except ValueError as exc:
        raise BenchmarkError(f"Некорректные числа RAW_FRAME: {lines[0]!r}") from exc

    records = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 9 or parts[0] != "S":
            raise BenchmarkError(f"Некорректная raw-строка: {line!r}")
        try:
            records.append(
                RawRecord(
                    host_time_s=host_time_s,
                    device_ms=device_ms,
                    sequence=sequence,
                    face=face,
                    sensor_id=int(parts[1]),
                    ax=int(parts[2]),
                    ay=int(parts[3]),
                    az=int(parts[4]),
                    temperature_raw=int(parts[5]),
                    gx=int(parts[6]),
                    gy=int(parts[7]),
                    gz=int(parts[8]),
                )
            )
        except ValueError as exc:
            raise BenchmarkError(f"Некорректные raw-данные: {line!r}") from exc
    if len(records) != expected_count:
        raise BenchmarkError(
            f"RAW_FRAME обещает {expected_count} датчиков, получено {len(records)}"
        )
    return records


def mean_accel(records: Sequence[RawRecord]) -> tuple[float, float, float]:
    return tuple(
        statistics.fmean((record.ax, record.ay, record.az)[axis] for record in records)
        for axis in range(3)
    )


def summarize_face(records: Sequence[RawRecord]) -> dict[str, object]:
    if not records:
        raise BenchmarkError("Нет raw-измерений")
    face = records[0].face
    sensor_id = records[0].sensor_id
    ordered = sorted(records, key=lambda record: record.host_time_s)
    mean = mean_accel(ordered)
    norms = [
        vector_norm((record.ax, record.ay, record.az)) / ACCEL_LSB_PER_G
        for record in ordered
    ]
    window = max(1, len(ordered) // 5)
    direction_drift = vector_angle_deg(
        mean_accel(ordered[:window]),
        mean_accel(ordered[-window:]),
    )
    clipped_components = sum(
        abs(value) >= 32760
        for record in ordered
        for value in (record.ax, record.ay, record.az)
    )
    clipped_fraction = clipped_components / (3.0 * len(ordered))
    gyro_rms = math.sqrt(
        statistics.fmean(
            record.gx * record.gx + record.gy * record.gy + record.gz * record.gz
            for record in ordered
        )
    ) / GYRO_LSB_PER_DPS
    face_error = vector_angle_deg(mean, EXPECTED[face])
    norm_mean = statistics.fmean(norms)
    norm_std = statistics.pstdev(norms)
    checks = {
        "samples": len(ordered) >= 10,
        "not_clipped": clipped_fraction == 0.0,
        "face_error": face_error <= 15.0,
        "norm": abs(norm_mean - 1.0) <= 0.20,
        "direction_drift": direction_drift <= 3.0,
        "gyro_stationary": gyro_rms <= 5.0,
    }
    return {
        "sensor_id": sensor_id,
        "face": face,
        "samples": len(ordered),
        "mean_accel_counts": {"x": mean[0], "y": mean[1], "z": mean[2]},
        "accel_norm_mean_g": norm_mean,
        "accel_norm_std_g": norm_std,
        "face_error_deg": face_error,
        "direction_drift_deg": direction_drift,
        "clipped_fraction": clipped_fraction,
        "gyro_rms_dps": gyro_rms,
        "checks": checks,
        "passed": all(checks.values()),
    }


def analyze(records: Sequence[RawRecord]) -> dict[str, object]:
    grouped: dict[tuple[int, str], list[RawRecord]] = {}
    for record in records:
        grouped.setdefault((record.sensor_id, record.face), []).append(record)
    summaries = [
        summarize_face(group)
        for _, group in sorted(
            grouped.items(),
            key=lambda item: (item[0][0], FACES.index(item[0][1])),
        )
    ]
    by_sensor: dict[int, dict[str, dict[str, object]]] = {}
    for summary in summaries:
        by_sensor.setdefault(int(summary["sensor_id"]), {})[str(summary["face"])] = summary
    z_calibration = []
    for sensor_id, faces in sorted(by_sensor.items()):
        if not all(face in faces for face in FACES):
            continue
        positive = float(faces["+Z"]["mean_accel_counts"]["z"])
        negative = float(faces["-Z"]["mean_accel_counts"]["z"])
        z_calibration.append(
            {
                "sensor_id": sensor_id,
                "z_bias_counts": (positive + negative) / 2.0,
                "z_scale_counts_per_g": (positive - negative) / 2.0,
            }
        )
    sensor_ids = sorted({record.sensor_id for record in records})
    complete = all(
        (sensor_id, face) in grouped for sensor_id in sensor_ids for face in FACES
    )
    return {
        "protocol": "RAW_FRAME/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "complete": complete,
        "passed": complete and all(bool(item["passed"]) for item in summaries),
        "face_summaries": summaries,
        "z_calibration": z_calibration,
    }


def build_report(summary: dict[str, object]) -> str:
    lines = [
        "# Raw MPU-6500: тест +Z/-Z без DMP",
        "",
        f"Итог: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        "| Датчик | Грань | N | ax | ay | az | Норма, g | Ошибка | Дрейф | Saturation | Гиро RMS |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["face_summaries"]:
        accel = item["mean_accel_counts"]
        lines.append(
            "| {sensor_id} | {face} {marker} | {samples} | {x:.0f} | {y:.0f} | "
            "{z:.0f} | {accel_norm_mean_g:.4f} | {face_error_deg:.2f}° | "
            "{direction_drift_deg:.2f}° | {clip:.2f}% | {gyro_rms_dps:.2f}°/s |".format(
                marker="✓" if item["passed"] else "✗",
                clip=100.0 * item["clipped_fraction"],
                **item,
                **accel,
            )
        )
    lines.extend(
        [
            "",
            "## Оценка оси Z",
            "",
            "| Датчик | Смещение Z, counts | Чувствительность Z, counts/g |",
            "|---:|---:|---:|",
        ]
    )
    for item in summary["z_calibration"]:
        lines.append(
            f"| {item['sensor_id']} | {item['z_bias_counts']:.1f} | "
            f"{item['z_scale_counts_per_g']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Ожидается около +16384 по Z в положении +Z и около -16384 в "
            "положении -Z. Этот тест не загружает DMP и не записывает offsets.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, records: Sequence[RawRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    os.replace(temporary, path)


def collect_face(
    udp_socket: socket.socket,
    target: tuple[str, int],
    face: str,
    duration_s: float,
    packet_timeout_s: float,
) -> list[RawRecord]:
    records: list[RawRecord] = []
    deadline = time.monotonic() + duration_s
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
            raise BenchmarkError("Поток RAW_FRAME прервался") from exc
        if address[0] == target[0]:
            records.extend(parse_raw_frame(payload, time.time(), face))
    if not records:
        raise BenchmarkError(f"Для {face} не получено raw-измерений")
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw MPU-6500 тест +Z/-Z без DMP")
    parser.add_argument("controller_ip")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--duration", type=positive_float, default=12.0)
    parser.add_argument("--settle", type=nonnegative_float, default=1.0)
    parser.add_argument("--command-timeout", type=positive_float, default=3.0)
    parser.add_argument("--packet-timeout", type=positive_float, default=2.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir or (
        Path(__file__).resolve().parent
        / "results_raw"
        / time.strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_samples.csv"
    target = (socket.gethostbyname(args.controller_ip), args.port)
    records: list[RawRecord] = []
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(("", 0))
    started = False
    try:
        hello = command(udp_socket, target, "HELLO", args.command_timeout)
        if "raw_benchmark=1" not in hello:
            raise BenchmarkError("На ESP32 запущена не raw-прошивка MPU-6500")
        sensors = re.search(r"\bsensors=(\d+)\b", hello)
        ready = re.search(r"\bready=(\d+)\b", hello)
        if sensors is None or ready is None or sensors.group(1) != ready.group(1):
            raise BenchmarkError(f"Не все raw-датчики готовы: {hello}")
        print(f"ESP32: {hello}")
        command(udp_socket, target, "START", args.command_timeout)
        started = True
        for face in FACES:
            input(f"\nУложите ВСЕ датчики: {DESCRIPTIONS[face]}. Нажмите Enter...")
            drain_socket(udp_socket)
            if args.settle:
                time.sleep(args.settle)
                drain_socket(udp_socket)
            face_records = collect_face(
                udp_socket,
                target,
                face,
                args.duration,
                args.packet_timeout,
            )
            records.extend(face_records)
            write_csv(raw_path, records)
            counts: dict[int, int] = {}
            for record in face_records:
                counts[record.sensor_id] = counts.get(record.sensor_id, 0) + 1
            print(
                "Записано: "
                + ", ".join(f"S{key}={value}" for key, value in sorted(counts.items()))
            )
    finally:
        if started:
            try:
                drain_socket(udp_socket)
                command(udp_socket, target, "STOP", args.command_timeout)
            except BenchmarkError as exc:
                print(f"Предупреждение STOP: {exc}", file=sys.stderr)
        udp_socket.close()

    summary = analyze(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(f"\nРезультат: {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"Отчёт: {output_dir / 'report.md'}")
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        print("\nТест прерван.", file=sys.stderr)
        return 130
    except (BenchmarkError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
