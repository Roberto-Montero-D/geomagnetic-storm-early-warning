"""Causal normalization utilities for the OMNI Kp index.

OMNI stores the 3-hour Kp index on each of the three hourly rows
belonging to that Kp interval.

For example:

    00:00  Kp = 50
    01:00  Kp = 50
    02:00  Kp = 50

represents one physical Kp interval:

    [00:00, 03:00)  Kp = 5.0

For prediction features, Kp must never be read directly from the
repeated hourly OMNI column. Instead:

1. Raw OMNI Kp is converted from the Kp*10 encoding.
2. Repeated hourly values are collapsed into canonical 3-hour intervals.
3. Kp(t-h) is interpreted through an as-of query on completed intervals.

The project's protocol cutoff is applied by querying:

    kp_asof(t - h)

so, for the most recent Kp feature:

    Kp(t-1h) = kp_asof(t - 1h)

This prevents within-interval future leakage.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


VALID_OMNI_KP_CODES = frozenset(
    [0, 3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37, 40, 43, 47,
     50, 53, 57, 60, 63, 67, 70, 73, 77, 80, 83, 87, 90]
)
OMNI_KP_FILL_VALUE = 99
DEFAULT_KP_LAGS_HOURS = (1, 3, 6, 12, 24)


def _validate_datetime_index(df: pd.DataFrame) -> None:
    """Validate basic timestamp requirements."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas DatetimeIndex.")
    if df.index.has_duplicates:
        raise ValueError("Kp input contains duplicate timestamps.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Kp timestamps must be monotonically increasing.")


def _validate_hourly_continuity(df: pd.DataFrame) -> None:
    """Require a continuous hourly source timeline."""
    if df.empty:
        return
    expected_index = pd.date_range(df.index[0], df.index[-1], freq="h")
    if not df.index.equals(expected_index):
        missing_timestamps = expected_index.difference(df.index)
        raise ValueError(
            "Kp input is not a continuous hourly time series. "
            f"Missing timestamp(s): {missing_timestamps.tolist()}."
        )


def convert_omni_kp_raw(
    series: pd.Series,
    *,
    fill_value: int = OMNI_KP_FILL_VALUE,
) -> pd.Series:
    """Convert raw OMNI Kp*10 encoding to standard Kp units."""
    numeric = pd.to_numeric(series, errors="raise").astype(float)
    numeric = numeric.mask(numeric == fill_value)
    non_missing = numeric.dropna()

    non_integer_mask = (non_missing % 1) != 0
    if non_integer_mask.any():
        invalid_values = sorted(non_missing[non_integer_mask].unique().tolist())
        raise ValueError(
            "OMNI Kp raw values must use integer Kp*10 codes. "
            f"Invalid value(s): {invalid_values}"
        )

    observed_codes = set(non_missing.astype(int).unique().tolist())
    invalid_codes = observed_codes - VALID_OMNI_KP_CODES
    if invalid_codes:
        raise ValueError(f"Invalid OMNI Kp code(s) found: {sorted(invalid_codes)}")

    return numeric / 10.0


def build_kp_intervals(
    df: pd.DataFrame,
    *,
    kp_column: str = "kp_raw",
) -> pd.DataFrame:
    """Collapse repeated hourly OMNI Kp into canonical 3-hour intervals.

    Validation semantics are intentionally strict and unchanged: the source
    must be hourly-contiguous, every canonical interval must contain exactly
    three rows, partial missingness is invalid, and the three repeated Kp
    values must agree. Fully missing intervals remain explicit NaN intervals.

    The successful path is vectorized so multi-decade OMNI inputs do not pay
    Python-loop overhead once per 3-hour interval.
    """
    if kp_column not in df.columns:
        raise KeyError(f"Missing required column: {kp_column!r}")

    _validate_datetime_index(df)
    _validate_hourly_continuity(df)

    if df.empty:
        return pd.DataFrame(columns=["interval_start", "interval_end", "kp"])

    kp = convert_omni_kp_raw(df[kp_column])
    starts = df.index.floor("3h")
    codes, unique_starts = pd.factorize(starts, sort=False)
    n_groups = len(unique_starts)

    counts = np.bincount(codes, minlength=n_groups)
    bad_size = np.flatnonzero(counts != 3)
    if bad_size.size:
        i = int(bad_size[0])
        interval_start = pd.Timestamp(unique_starts[i])
        expected = pd.date_range(interval_start, periods=3, freq="h")
        got = df.index[codes == i]
        raise ValueError(
            "Incomplete or non-contiguous Kp interval starting at "
            f"{interval_start}. Expected timestamps: {expected.tolist()}, "
            f"got: {got.tolist()}."
        )

    values = kp.to_numpy(dtype=float)
    missing = np.isnan(values)
    missing_counts = np.bincount(codes, weights=missing.astype(np.int8), minlength=n_groups)
    partial = np.flatnonzero((missing_counts > 0) & (missing_counts < 3))
    if partial.size:
        interval_start = pd.Timestamp(unique_starts[int(partial[0])])
        raise ValueError(f"Partially missing Kp interval starting at {interval_start}.")

    # Because every valid group has exactly three consecutive rows, reshape is
    # safe and substantially cheaper than a Python groupby loop.
    matrix = values.reshape(-1, 3)
    all_missing = np.isnan(matrix).all(axis=1)
    disagreement = (~all_missing) & (
        (matrix[:, 0] != matrix[:, 1]) | (matrix[:, 0] != matrix[:, 2])
    )
    if disagreement.any():
        i = int(np.flatnonzero(disagreement)[0])
        interval_start = pd.Timestamp(unique_starts[i])
        unique_values = pd.unique(matrix[i]).tolist()
        raise ValueError(
            "Inconsistent repeated Kp values in interval "
            f"starting at {interval_start}: {unique_values}."
        )

    canonical = matrix[:, 0].copy()
    canonical[all_missing] = np.nan
    interval_start_index = pd.DatetimeIndex(unique_starts)

    return pd.DataFrame(
        {
            "interval_start": interval_start_index,
            "interval_end": interval_start_index + pd.Timedelta(hours=3),
            "kp": canonical,
        }
    )


def kp_asof(
    intervals: pd.DataFrame,
    query_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.Series:
    """Return Kp from the most recent interval completed by each query time."""
    required = {"interval_end", "kp"}
    missing = required - set(intervals.columns)
    if missing:
        raise KeyError(f"Missing interval column(s): {sorted(missing)}")

    query_index = pd.DatetimeIndex(query_times)
    if intervals.empty:
        return pd.Series(np.nan, index=query_index, dtype=float, name="kp")

    ordered = intervals.sort_values("interval_end").reset_index(drop=True)
    ends = pd.DatetimeIndex(ordered["interval_end"])
    kp_values = ordered["kp"].to_numpy(dtype=float)
    positions = ends.searchsorted(query_index, side="right") - 1
    result = np.full(len(query_index), np.nan, dtype=float)
    valid = positions >= 0
    result[valid] = kp_values[positions[valid]]
    return pd.Series(result, index=query_index, name="kp")


def build_kp_lag_features(
    intervals: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    lags_hours: tuple[int, ...] = DEFAULT_KP_LAGS_HOURS,
) -> pd.DataFrame:
    """Build causal Kp lag features for prediction timestamps."""
    prediction_index = pd.DatetimeIndex(prediction_times)
    if any(not isinstance(lag, int) or lag <= 0 for lag in lags_hours):
        raise ValueError("All Kp lags must be positive integers.")
    if len(set(lags_hours)) != len(lags_hours):
        raise ValueError("Kp lag values must be unique.")

    features = pd.DataFrame(index=prediction_index)
    for lag in lags_hours:
        query_times = prediction_index - pd.Timedelta(hours=lag)
        values = kp_asof(intervals, query_times)
        features[f"kp_lag_{lag}h"] = values.to_numpy()
    return features
