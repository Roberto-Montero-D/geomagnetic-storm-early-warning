"""Real-data smoke test for OMNI ingestion and causal Kp normalization."""

from pathlib import Path

import pandas as pd

from src.data.kp import (
    build_kp_intervals,
    build_kp_lag_features,
)
from src.data.omni import load_omni


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

FMT_PATH = Path("data/raw/omni.fmt")
LST_PATH = Path("data/raw/omni.lst")


# ---------------------------------------------------------------------
# Load real OMNI data
# ---------------------------------------------------------------------

print("Loading OMNI data...")

omni = load_omni(
    FMT_PATH,
    LST_PATH,
)

print("\n=== OMNI ===")
print(f"Rows: {len(omni):,}")
print(f"Start: {omni.index.min()}")
print(f"End:   {omni.index.max()}")
print(f"Columns ({len(omni.columns)}):")
print(list(omni.columns))

print("\nFirst 3 rows:")
print(omni.head(3))

print("\nLast 3 rows:")
print(omni.tail(3))


# ---------------------------------------------------------------------
# Basic timeline checks
# ---------------------------------------------------------------------

expected_index = pd.date_range(
    start=omni.index.min(),
    end=omni.index.max(),
    freq="h",
)

assert omni.index.equals(expected_index), (
    "OMNI timeline is not continuous hourly data."
)

assert omni.index.is_unique
assert omni.index.is_monotonic_increasing

print("\nHourly timeline: PASS")


# ---------------------------------------------------------------------
# Kp interval construction
# ---------------------------------------------------------------------

print("\nBuilding canonical Kp intervals...")

intervals = build_kp_intervals(
    omni[["kp_raw"]],
)

print("\n=== Kp INTERVALS ===")
print(f"Intervals: {len(intervals):,}")
print(
    f"First interval: "
    f"{intervals.iloc[0]['interval_start']} -> "
    f"{intervals.iloc[0]['interval_end']}"
)
print(
    f"Last interval:  "
    f"{intervals.iloc[-1]['interval_start']} -> "
    f"{intervals.iloc[-1]['interval_end']}"
)

print("\nFirst 5 intervals:")
print(intervals.head())

print("\nLast 5 intervals:")
print(intervals.tail())


# ---------------------------------------------------------------------
# Structural Kp checks
# ---------------------------------------------------------------------

expected_intervals = len(omni) // 3

assert len(omni) % 3 == 0, (
    "Hourly OMNI row count is not divisible by 3."
)

assert len(intervals) == expected_intervals, (
    f"Expected {expected_intervals:,} Kp intervals, "
    f"got {len(intervals):,}."
)

assert (
    intervals["interval_end"]
    - intervals["interval_start"]
    == pd.Timedelta(hours=3)
).all()

print("\nKp interval structure: PASS")


# ---------------------------------------------------------------------
# Missing Kp
# ---------------------------------------------------------------------

missing_kp = intervals["kp"].isna().sum()

print(f"\nMissing canonical Kp intervals: {missing_kp:,}")


# ---------------------------------------------------------------------
# Causal lag generation
# ---------------------------------------------------------------------

print("\nBuilding causal Kp lag features...")

kp_features = build_kp_lag_features(
    intervals,
    omni.index,
)

print("\n=== Kp FEATURES ===")
print(f"Rows: {len(kp_features):,}")
print(f"Columns: {list(kp_features.columns)}")

print("\nFirst 10 rows:")
print(kp_features.head(10))

print("\nLast 5 rows:")
print(kp_features.tail())

print("\nMissing values by feature:")
print(kp_features.isna().sum())


# ---------------------------------------------------------------------
# Explicit causality spot check
# ---------------------------------------------------------------------

# Choose a timestamp safely away from the beginning of the dataset.
t = omni.index[100]

row = kp_features.loc[t]

print("\n=== CAUSAL SPOT CHECK ===")
print(f"Prediction time: {t}")

for lag in (1, 3, 6, 12, 24):
    query_time = t - pd.Timedelta(hours=lag)

    eligible = intervals[
        intervals["interval_end"] <= query_time
    ]

    expected = (
        eligible.iloc[-1]["kp"]
        if not eligible.empty
        else float("nan")
    )

    actual = row[f"kp_lag_{lag}h"]

    if pd.isna(expected):
        assert pd.isna(actual)
    else:
        assert actual == expected

    print(
        f"lag={lag:>2}h | "
        f"query={query_time} | "
        f"Kp={actual}"
    )

print("\nCausal spot check: PASS")


# ---------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------

print("\n======================================")
print("REAL OMNI -> Kp SMOKE TEST PASSED")
print("======================================")