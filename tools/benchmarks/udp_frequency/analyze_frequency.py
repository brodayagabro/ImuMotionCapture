#!/usr/bin/env python3
"""Calculate packet-arrival statistics and save them as CSV."""

from pathlib import Path

import numpy as np
import pandas as pd


script_directory = Path(__file__).resolve().parent
input_path = script_directory / "frequency_timestamps.csv"
statistics_path = script_directory / "frequency_analysis.csv"

# Each CSV column contains packet-arrival timestamps for one target frequency.
timestamps = pd.read_csv(input_path).apply(pd.to_numeric, errors="raise")
skipped_targets = timestamps.columns[timestamps.count() < 2]
timestamps = timestamps.loc[:, timestamps.count() >= 2]

if timestamps.empty:
    raise ValueError("ни в одном столбце нет хотя бы двух временных меток")

# Drop empty cells in each column first, then calculate adjacent intervals.
intervals_s = timestamps.apply(
    lambda column: pd.Series(np.diff(column.dropna().to_numpy()))
)
if intervals_s.le(0).any().any():
    invalid_targets = intervals_s.columns[intervals_s.le(0).any()]
    invalid_list = ", ".join(invalid_targets)
    raise ValueError(f"метки для {invalid_list} Гц не строго возрастают")

target_hz = pd.to_numeric(timestamps.columns).to_numpy(dtype=float)

mean_interval_s = intervals_s.mean().to_numpy()

actual_hz = 1.0 / mean_interval_s

statistics = pd.DataFrame(
    {
        "target_frequency_hz": target_hz,
        "timestamp_count": timestamps.count().to_numpy(),
        "interval_count": intervals_s.count().to_numpy(),
        "mean_interval_s": mean_interval_s,
        "mean_interval_ms": mean_interval_s * 1e3,
        "actual_frequency_hz": actual_hz,
    }
)
statistics.to_csv(statistics_path, index=False, float_format="%.12g")

print(
    statistics[
        [
            "target_frequency_hz",
            "actual_frequency_hz"
        ]
    ].to_string(index=False)
)
if len(skipped_targets):
    print(f"Пропущены пустые столбцы: {', '.join(skipped_targets)} Гц")
print(f"Статистика: {statistics_path}")
