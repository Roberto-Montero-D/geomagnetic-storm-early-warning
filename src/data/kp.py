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


# OMNI Kp integer encoding:
# 0, 0+, 1-, 1, 1+, ..., 9
VALID_OMNI_KP_CODES = frozenset(
    [
        0,
        3,
        7,
        10,
        13,
        17,
        20,
        23,
        27,
        30,
        33,
        37,
        40,
        43,
        47,
        50,
        53,
        57,
        60,
        63,
        67,
        70,
        73,
        77,
        80,
        83,
        87,
        90,
    ]
)

# OMNI fill value for unavailable Kp.
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


def convert_omni_kp_raw(
    series: pd.Series,
    *,
    fill_value: int = OMNI_KP_FILL_VALUE,
) -> pd.Series:
    """Convert raw OMNI Kp*10 encoding to standard Kp units.

    Parameters
    ----------
    series
        Raw OMNI Kp values.
    fill_value
        OMNI missing-value sentinel. Default is 99.

    Returns
    -------
    pandas.Series
        Standard Kp values such as 5.0 instead of raw 50.

    Raises
    ------
    ValueError
        If a non-missing raw value is not a valid OMNI Kp code.
    """

    numeric = pd.to_numeric(series, errors="raise").astype(float)

    numeric = numeric.mask(numeric == fill_value)

    observed_codes = set(
        numeric.dropna().astype(int).unique().tolist()
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

    The raw timestamp is treated as the start of each hourly row.
    Kp intervals are aligned to:

        00-03
        03-06
        06-09
        ...
        21-24 UTC

    Every canonical interval must contain exactly three consecutive
    hourly rows.

    Missing Kp intervals remain missing. They are not forward-filled.

    Returns
    -------
    pandas.DataFrame
        Columns:

        interval_start
        interval_end
        kp

    Raises
    ------
    ValueError
        If the source does not contain exactly three hourly rows per
        3-hour Kp interval, or repeated Kp values disagree.
    """

    if kp_column not in df.columns:
        raise KeyError(f"Missing required column: {kp_column!r}")

    _validate_datetime_index(df)

    work = df[[kp_column]].copy()
    work["kp"] = convert_omni_kp_raw(work[kp_column])

    interval_starts = work.index.floor("3h")
    work["interval_start"] = interval_starts

    records: list[dict[str, object]] = []

    for interval_start, group in work.groupby(
        "interval_start",
        sort=True,
    ):
        expected_timestamps = pd.date_range(
            start=interval_start,
            periods=3,
            freq="h",
        )

        if len(group) != 3 or not group.index.equals(expected_timestamps):
            raise ValueError(
                "Incomplete or non-contiguous Kp interval starting at "
                f"{interval_start}. Expected timestamps: "
                f"{expected_timestamps.tolist()}, got: "
                f"{group.index.tolist()}."
            )

        values = group["kp"]

        if values.isna().all():
            kp_value = np.nan

        elif values.isna().any():
            raise ValueError(
                "Partially missing Kp interval starting at "
                f"{interval_start}."
            )

        else:
            unique_values = values.unique()

            if len(unique_values) != 1:
                raise ValueError(
                    "Inconsistent repeated Kp values in interval "
                    f"starting at {interval_start}: "
                    f"{unique_values.tolist()}."
                )

            kp_value = float(unique_values[0])

        records.append(
            {
                "interval_start": interval_start,
                "interval_end": interval_start
                + pd.Timedelta(hours=3),
                "kp": kp_value,
            }
        )

    return pd.DataFrame.from_records(records)


def kp_asof(
    intervals: pd.DataFrame,
    query_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.Series:
    """Return Kp from the most recent interval completed by each query time.

    For query time q:

        kp_asof(q)

    returns Kp from the most recent interval satisfying:

        interval_end <= q

    Importantly, if that most recent completed interval has missing Kp,
    the result is missing. The function does NOT skip backward to an
    older valid interval.

    Parameters
    ----------
    intervals
        Canonical output from ``build_kp_intervals``.
    query_times
        Times at which causal Kp state should be evaluated.

    Returns
    -------
    pandas.Series
        Kp values indexed by query time.
    """

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

    ordered = intervals.sort_values("interval_end").reset_index(drop=True)

    ends = pd.DatetimeIndex(ordered["interval_end"])
    kp_values = ordered["kp"].to_numpy(dtype=float)

    # searchsorted(..., side="right") - 1 gives the most recent
    # interval satisfying interval_end <= query_time.
    positions = ends.searchsorted(query_index, side="right") - 1

    result = np.full(len(query_index), np.nan, dtype=float)

    valid = positions >= 0
    result[valid] = kp_values[positions[valid]]

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
    """Build causal Kp lag features for prediction timestamps.

    The canonical definition is:

        Kp_lag_h(t) = kp_asof(t - h)

    Therefore:

        Kp_lag_1h(t) = kp_asof(t - 1h)

    The protocol's one-hour causal cutoff is applied exactly once.
    """

    prediction_index = pd.DatetimeIndex(prediction_times)

    if any(lag <= 0 for lag in lags_hours):
        raise ValueError("All Kp lags must be positive integers.")

    features = pd.DataFrame(index=prediction_index)

    for lag in lags_hours:
        query_times = prediction_index - pd.Timedelta(hours=lag)

        values = kp_asof(
            intervals,
            query_times,
        )

        # kp_asof is indexed by query time; feature rows must remain
        # indexed by prediction time.
        features[f"kp_lag_{lag}h"] = values.to_numpy()

    return features