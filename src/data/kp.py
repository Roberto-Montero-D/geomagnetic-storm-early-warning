"""Causal normalization utilities for the OMNI Kp index."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


VALID_OMNI_KP_CODES = frozenset(
    [
        0, 3, 7, 10, 13, 17, 20, 23, 27, 30,
        33, 37, 40, 43, 47, 50, 53, 57, 60, 63,
        67, 70, 73, 77, 80, 83, 87, 90,
    ]
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

    index = df.index

    if len(index) <= 1:
        return

    # Normalize explicitly to nanoseconds before integer arithmetic.
    # Pandas 3 may preserve datetime64[us] indexes, so comparing ``asi8``
    # directly with Timedelta.value (nanoseconds) is resolution-dependent.
    index_ns = index.to_numpy(dtype="datetime64[ns]").astype("int64")
    deltas = np.diff(index_ns)
    expected_delta = np.timedelta64(1, "h").astype("timedelta64[ns]").astype("int64")

    if np.all(deltas == expected_delta):
        return

    expected_index = pd.date_range(
        start=index[0],
        end=index[-1],
        freq="h",
    )
    missing_timestamps = expected_index.difference(index)

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
        invalid_values = sorted(
            non_missing[non_integer_mask].unique().tolist()
        )
        raise ValueError(
            "OMNI Kp raw values must use integer Kp*10 codes. "
            f"Invalid value(s): {invalid_values}"
        )

    observed_codes = set(
        non_missing.astype(int).unique().tolist()
    )
    invalid_codes = observed_codes - VALID_OMNI_KP_CODES

    if invalid_codes:
        raise ValueError(
            "Invalid OMNI Kp code(s) found: "
            f"{sorted(invalid_codes)}"
        )

    return numeric / 10.0


def build_kp_intervals(
    df: pd.DataFrame,
    *,
    kp_column: str = "kp_raw",
) -> pd.DataFrame:
    """Collapse repeated hourly OMNI Kp into canonical 3-hour intervals.

    This is the vectorized equivalent of the original group-by implementation.
    The validation and missing-value semantics remain unchanged.
    """
    if kp_column not in df.columns:
        raise KeyError(f"Missing required column: {kp_column!r}")

    _validate_datetime_index(df)
    _validate_hourly_continuity(df)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "interval_start",
                "interval_end",
                "kp",
            ]
        )

    kp = convert_omni_kp_raw(df[kp_column])
    index = df.index

    interval_start = index.floor("3h")

    # A valid source must begin and end on complete canonical 3-hour blocks.
    # factorize preserves the chronological group order because the input
    # index is already validated as monotonic.
    codes, unique_starts = pd.factorize(
        interval_start,
        sort=False,
    )

    group_count = len(unique_starts)
    counts = np.bincount(
        codes,
        minlength=group_count,
    )

    invalid_size = np.flatnonzero(counts != 3)
    if invalid_size.size:
        group_id = int(invalid_size[0])
        start = pd.Timestamp(unique_starts[group_id])
        expected_timestamps = pd.date_range(
            start=start,
            periods=3,
            freq="h",
        )
        actual_timestamps = index[codes == group_id]

        raise ValueError(
            "Incomplete or non-contiguous Kp interval starting at "
            f"{start}. Expected timestamps: "
            f"{expected_timestamps.tolist()}, got: "
            f"{actual_timestamps.tolist()}."
        )

    values = kp.to_numpy(dtype=float, copy=False)

    missing = np.isnan(values)
    missing_counts = np.bincount(
        codes,
        weights=missing.astype(np.int8),
        minlength=group_count,
    )

    partially_missing = np.flatnonzero(
        (missing_counts > 0)
        & (missing_counts < 3)
    )

    if partially_missing.size:
        group_id = int(partially_missing[0])
        start = pd.Timestamp(unique_starts[group_id])

        raise ValueError(
            "Partially missing Kp interval starting at "
            f"{start}."
        )

    # Hourly continuity plus exactly three rows per group guarantees that
    # each canonical group occupies three adjacent positions.
    matrix = values.reshape(group_count, 3)

    all_missing = np.isnan(matrix).all(axis=1)

    inconsistent = (
        ~all_missing
        & (
            (matrix[:, 0] != matrix[:, 1])
            | (matrix[:, 0] != matrix[:, 2])
        )
    )

    if inconsistent.any():
        group_id = int(
            np.flatnonzero(inconsistent)[0]
        )
        start = pd.Timestamp(unique_starts[group_id])

        unique_values = pd.unique(
            matrix[group_id]
        ).tolist()

        raise ValueError(
            "Inconsistent repeated Kp values in interval "
            f"starting at {start}: "
            f"{unique_values}."
        )

    canonical_kp = matrix[:, 0].copy()
    canonical_kp[all_missing] = np.nan

    starts = pd.DatetimeIndex(unique_starts)

    return pd.DataFrame(
        {
            "interval_start": starts,
            "interval_end": (
                starts
                + pd.Timedelta(hours=3)
            ),
            "kp": canonical_kp,
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
        raise KeyError(
            f"Missing interval column(s): {sorted(missing)}"
        )

    query_index = pd.DatetimeIndex(query_times)

    if intervals.empty:
        return pd.Series(
            np.nan,
            index=query_index,
            dtype=float,
            name="kp",
        )

    # Canonical interval tables are normally already sorted. Avoid an
    # unnecessary full sort on the common path while retaining compatibility
    # with arbitrary callers.
    interval_end = pd.DatetimeIndex(
        intervals["interval_end"]
    )

    if interval_end.is_monotonic_increasing:
        ordered = intervals
    else:
        ordered = intervals.sort_values(
            "interval_end"
        ).reset_index(drop=True)
        interval_end = pd.DatetimeIndex(
            ordered["interval_end"]
        )

    kp_values = ordered["kp"].to_numpy(
        dtype=float,
        copy=False,
    )

    positions = interval_end.searchsorted(
        query_index,
        side="right",
    ) - 1

    result = np.full(
        len(query_index),
        np.nan,
        dtype=float,
    )

    valid = positions >= 0
    result[valid] = kp_values[
        positions[valid]
    ]

    return pd.Series(
        result,
        index=query_index,
        name="kp",
    )


def build_kp_lag_features(
    intervals: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    lags_hours: tuple[int, ...] = DEFAULT_KP_LAGS_HOURS,
) -> pd.DataFrame:
    """Build causal Kp lag features for prediction timestamps."""
    prediction_index = pd.DatetimeIndex(
        prediction_times
    )

    if any(
        not isinstance(lag, int) or lag <= 0
        for lag in lags_hours
    ):
        raise ValueError(
            "All Kp lags must be positive integers."
        )

    if len(set(lags_hours)) != len(lags_hours):
        raise ValueError(
            "Kp lag values must be unique."
        )

    features = pd.DataFrame(
        index=prediction_index
    )

    for lag in lags_hours:
        query_times = (
            prediction_index
            - pd.Timedelta(hours=lag)
        )

        values = kp_asof(
            intervals,
            query_times,
        )

        features[
            f"kp_lag_{lag}h"
        ] = values.to_numpy()

    return features
