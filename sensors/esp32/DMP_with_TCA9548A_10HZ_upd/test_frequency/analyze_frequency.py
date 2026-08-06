#!/usr/bin/env python3
"""Calculate packet-arrival statistics and plot measured versus target rate."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class FrequencyStats:
    target_hz: float
    timestamp_count: int
    interval_count: int
    mean_interval_s: float
    interval_variance_s2: float
    interval_std_s: float
    actual_frequency_hz: float
    error_hz: float
    relative_error_percent: float


def read_statistics(input_path: Path) -> tuple[list[FrequencyStats], list[float]]:
    """Read one target frequency per column and calculate adjacent deltas."""
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))

    if not rows:
        raise ValueError("CSV-файл пуст")

    try:
        targets_hz = [float(value) for value in rows[0]]
    except ValueError as error:
        raise ValueError("заголовок CSV должен содержать целевые частоты") from error

    results: list[FrequencyStats] = []
    skipped_targets: list[float] = []

    for column_index, target_hz in enumerate(targets_hz):
        timestamps: list[float] = []
        for row_number, row in enumerate(rows[1:], start=2):
            if column_index >= len(row) or not row[column_index].strip():
                continue
            try:
                timestamps.append(float(row[column_index]))
            except ValueError as error:
                raise ValueError(
                    f"нечисловая метка в строке {row_number}, "
                    f"столбце {column_index + 1}"
                ) from error

        if len(timestamps) < 2:
            skipped_targets.append(target_hz)
            continue

        intervals_s = [
            current - previous
            for previous, current in zip(timestamps, timestamps[1:])
        ]
        if any(interval <= 0.0 for interval in intervals_s):
            raise ValueError(f"метки для {target_hz:g} Гц не строго возрастают")

        mean_interval_s = statistics.fmean(intervals_s)
        # Population variance: all intervals available in the recording are used.
        interval_variance_s2 = statistics.pvariance(intervals_s)
        actual_frequency_hz = 1.0 / mean_interval_s
        error_hz = actual_frequency_hz - target_hz

        results.append(
            FrequencyStats(
                target_hz=target_hz,
                timestamp_count=len(timestamps),
                interval_count=len(intervals_s),
                mean_interval_s=mean_interval_s,
                interval_variance_s2=interval_variance_s2,
                interval_std_s=math.sqrt(interval_variance_s2),
                actual_frequency_hz=actual_frequency_hz,
                error_hz=error_hz,
                relative_error_percent=100.0 * error_hz / target_hz,
            )
        )

    if not results:
        raise ValueError("ни в одном столбце нет хотя бы двух временных меток")
    return results, skipped_targets


def save_statistics(output_path: Path, results: list[FrequencyStats]) -> None:
    """Save results in SI units and convenient millisecond units."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            (
                "target_frequency_hz",
                "timestamp_count",
                "interval_count",
                "mean_interval_s",
                "mean_interval_ms",
                "interval_variance_s2",
                "interval_variance_ms2",
                "interval_std_s",
                "interval_std_ms",
                "actual_frequency_hz",
                "frequency_error_hz",
                "relative_error_percent",
            )
        )
        for result in results:
            writer.writerow(
                (
                    f"{result.target_hz:.9g}",
                    result.timestamp_count,
                    result.interval_count,
                    f"{result.mean_interval_s:.12g}",
                    f"{result.mean_interval_s * 1e3:.12g}",
                    f"{result.interval_variance_s2:.12g}",
                    f"{result.interval_variance_s2 * 1e6:.12g}",
                    f"{result.interval_std_s:.12g}",
                    f"{result.interval_std_s * 1e3:.12g}",
                    f"{result.actual_frequency_hz:.12g}",
                    f"{result.error_hz:.12g}",
                    f"{result.relative_error_percent:.12g}",
                )
            )


def save_plot(output_path: Path, results: list[FrequencyStats]) -> None:
    """Plot actual rate, its error, and interval variance."""
    targets = [result.target_hz for result in results]
    actual = [result.actual_frequency_hz for result in results]
    errors = [result.error_hz for result in results]
    variances_ms2 = [result.interval_variance_s2 * 1e6 for result in results]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 13),
        sharex=True,
        constrained_layout=True,
    )
    figure.suptitle(
        "Частота прихода UDP-пакетов",
        fontsize=18,
        fontweight="bold",
    )

    axes[0].plot(
        targets,
        targets,
        "--",
        color="#8b96a8",
        linewidth=1.8,
        label="идеальная y = x",
    )
    axes[0].plot(
        targets,
        actual,
        "o-",
        color="#1664c0",
        linewidth=2.2,
        markersize=5,
        label="измеренная",
    )
    axes[0].set_ylabel("Действительная частота, Гц")
    axes[0].set_title("Частота = 1 / средний соседний интервал")
    axes[0].legend()

    axes[1].axhline(0.0, color="#596273", linewidth=1.2)
    axes[1].plot(
        targets,
        errors,
        "o-",
        color="#d04b32",
        linewidth=2.2,
        markersize=5,
    )
    axes[1].set_ylabel("Ошибка частоты, Гц")
    axes[1].set_title("Действительная минус целевая частота")

    axes[2].semilogy(
        targets,
        variances_ms2,
        "o-",
        color="#16856b",
        linewidth=2.2,
        markersize=5,
    )
    axes[2].set_ylabel("Дисперсия интервала, мс²")
    axes[2].set_xlabel("Целевая частота, Гц")
    axes[2].set_title("Дисперсия соседних интервалов (логарифмическая шкала)")

    for axis in axes:
        axis.grid(True, which="major", alpha=0.55)
        axis.set_xlim(min(targets) - 2.0, max(targets) + 2.0)
    axes[2].grid(True, which="minor", alpha=0.18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Вычислить соседние интервалы прихода пакетов, их среднее и "
            "дисперсию, пересчитать средний интервал в частоту и построить графики."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=script_directory / "frequency_timestamps.csv",
        help="таблица временных меток CSV",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=script_directory / "frequency_analysis.csv",
        help="выходная таблица статистики",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=script_directory / "frequency_analysis.png",
        help="выходное изображение с графиками",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    statistics_path = args.statistics.expanduser().resolve()
    plot_path = args.plot.expanduser().resolve()

    try:
        results, skipped_targets = read_statistics(input_path)
        save_statistics(statistics_path, results)
        save_plot(plot_path, results)
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    print("Цель, Гц | Действит., Гц | Средний интервал, мс | Дисперсия, мс²")
    for result in results:
        print(
            f"{result.target_hz:9g} | {result.actual_frequency_hz:13.3f} | "
            f"{result.mean_interval_s * 1e3:20.3f} | "
            f"{result.interval_variance_s2 * 1e6:15.3f}"
        )
    if skipped_targets:
        skipped = ", ".join(f"{value:g}" for value in skipped_targets)
        print(f"Пропущены пустые столбцы: {skipped} Гц")
    print(f"Статистика: {statistics_path}")
    print(f"Графики:    {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
