#!/usr/bin/env python3
"""Calculate packet-arrival statistics and plot measured versus target rate."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


script_directory = Path(__file__).resolve().parent
input_path = script_directory / "frequency_timestamps.csv"
statistics_path = script_directory / "frequency_analysis.csv"
plot_path = script_directory / "frequency_analysis.png"

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

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), sharex=True, layout="constrained")
ax.plot(target_hz, target_hz, "--")
statistics.plot(
    x="target_frequency_hz",
    y="actual_frequency_hz",
    style="o-",
    color='r',
    ax=ax,
)
fig.suptitle("Частота прихода UDP-пакетов", fontsize=18, fontweight="bold")
ax.set(title="Действительная частота", ylabel="Частота, Гц")
ax.legend(["идеальная y = x", "измеренная"])
fig.savefig(plot_path, dpi=180)
plt.close(fig)

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
print(f"Графики:    {plot_path}")
