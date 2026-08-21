"""Canonical retrospective Event-in-Window target construction.

The frozen primary target is:

    y_event(t) = max(Kp[t+1 : t+H]) >= T

with:
    T = 5
    H = 6 hours

Despite the historical name ``y_event``, the target is positive when
geomagnetic storm CONDITIONS (Kp >= T) occur anywhere in the future window.
It is not restricted to storm-onset timestamps.

The target uses retrospective canonical Kp ground truth and is intentionally
separate from predictor-side ``kp_asof()`` availability semantics.

Unknown future ground truth is handled conservatively:
- if any valid future state satisfies Kp >= T, target = 1;
- if all H future hourly states are valid and below T, target = 0;
- otherwise target = NaN.

Therefore incomplete right-edge horizons and missing Kp states are never
silently converted into negatives.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_TARGET_THRESHOLD = 5.0
DEFAULT_TARGET_HORIZON_HOURS = 6

_REQUIRED_INTERVAL_COLUMNS = {
    "interval_start",
    "interval_end",
    "kp",
}


def _validate_parameters(
    threshold: float,
    horizon_hours: int,
) -> tuple[float, int]:
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError("threshold must be numeric.") from exc

    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")

    if (
        not isinstance(horizon_hours, int)
        or isinstance(horizon_hours, bool)
    ):
        raise TypeError("horizon_hours must be an integer.")

    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be greater than zero.")

    return threshold_value, horizon_hours


def _validate_prediction_times(
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(
        prediction_times,
        name="prediction_time",
    )

    if index.hasnans:
        raise ValueError("prediction_times must not contain NaT.")

    if index.has_duplicates:
        raise ValueError("prediction_times must be unique.")

    if not index.is_monotonic_increasing:
        raise ValueError(
            "prediction_times must be monotonically increasing."
        )

    if len(index) and (
        (index.minute != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (index.nanosecond != 0).any()
    ):
        raise ValueError(
            "prediction_times must be aligned to whole hours."
        )

    return index


def _validate_intervals(
    kp_intervals: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(kp_intervals, pd.DataFrame):
        raise TypeError("kp_intervals must be a pandas DataFrame.")

    missing = (
        _REQUIRED_INTERVAL_COLUMNS
        - set(kp_intervals.columns)
    )
    if missing:
        raise ValueError(
            "kp_intervals is missing required columns: "
            f"{sorted(missing)}"
        )

    frame = kp_intervals[
        ["interval_start", "interval_end", "kp"]
    ].copy()

    frame["interval_start"] = pd.to_datetime(
        frame["interval_start"],
        errors="raise",
    )
    frame["interval_end"] = pd.to_datetime(
        frame["interval_end"],
        errors="raise",
    )

    if frame["interval_start"].isna().any():
        raise ValueError("interval_start must not contain NaT.")

    if frame["interval_end"].isna().any():
        raise ValueError("interval_end must not contain NaT.")

    if frame["interval_start"].duplicated().any():
        raise ValueError(
            "interval_start contains duplicate timestamps."
        )

    if not frame["interval_start"].is_monotonic_increasing:
        raise ValueError(
            "Kp intervals must be ordered by interval_start."
        )

    durations = (
        frame["interval_end"]
        - frame["interval_start"]
    )

    if not durations.eq(
        pd.Timedelta(hours=3)
    ).all():
        raise ValueError(
            "Every canonical Kp interval must span exactly 3 hours."
        )

    if len(frame) > 1:
        starts = frame["interval_start"].iloc[
            1:
        ].reset_index(drop=True)
        previous_ends = frame["interval_end"].iloc[
            :-1
        ].reset_index(drop=True)

        if (starts < previous_ends).any():
            raise ValueError(
                "Canonical Kp intervals must not overlap."
            )

    frame["kp"] = pd.to_numeric(
        frame["kp"],
        errors="raise",
    ).astype(float)

    invalid = (
        frame["kp"].notna()
        & ~np.isfinite(frame["kp"])
    )
    if invalid.any():
        raise ValueError(
            "kp must contain only finite values or NaN."
        )

    return frame.reset_index(drop=True)


def _expand_retrospective_kp_hourly(
    intervals: pd.DataFrame,
) -> pd.Series:
    """Expand canonical 3-hour Kp intervals onto hourly ground truth.

    Temporal gaps between intervals remain NaN.
    """

    if intervals.empty:
        return pd.Series(
            dtype=float,
            index=pd.DatetimeIndex(
                [],
                name="timestamp",
            ),
            name="kp",
        )

    first_start = intervals["interval_start"].iloc[0]
    final_end = intervals["interval_end"].iloc[-1]

    index = pd.date_range(
        first_start,
        final_end,
        freq="h",
        inclusive="left",
        name="timestamp",
    )

    hourly = pd.Series(
        np.nan,
        index=index,
        dtype=float,
        name="kp",
    )

    for row in intervals.itertuples(index=False):
        interval_hours = pd.date_range(
            row.interval_start,
            row.interval_end,
            freq="h",
            inclusive="left",
        )
        hourly.loc[interval_hours] = row.kp

    return hourly


def build_event_window_target(
    kp_intervals: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    threshold: float = DEFAULT_TARGET_THRESHOLD,
    horizon_hours: int = DEFAULT_TARGET_HORIZON_HOURS,
    return_audit: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Build the canonical retrospective future-Kp target.

    For prediction time ``t`` the exact hourly target window is:

        t + 1h, t + 2h, ..., t + H

    which implements the frozen inclusive interval:

        (t, t + H]

    Target semantics:
        1.0 -> at least one valid future hourly state has Kp >= threshold.
        0.0 -> all H future hourly states are valid and all are below threshold.
        NaN -> no positive state is observed, but one or more required future
               states are unavailable.

    A known positive is sufficient even if another hour in the window is
    missing, because the existential target condition has already been
    satisfied.
    """

    threshold, horizon_hours = _validate_parameters(
        threshold,
        horizon_hours,
    )
    prediction_index = _validate_prediction_times(
        prediction_times
    )
    intervals = _validate_intervals(kp_intervals)
    hourly = _expand_retrospective_kp_hourly(intervals)

    target = pd.Series(
        np.nan,
        index=prediction_index,
        dtype=float,
        name="target",
    )

    audit = pd.DataFrame(index=prediction_index)
    audit.index.name = "prediction_time"
    audit["future_window_start"] = (
        prediction_index + pd.Timedelta(hours=1)
    )
    audit["future_window_end"] = (
        prediction_index
        + pd.Timedelta(hours=horizon_hours)
    )
    audit["expected_future_hours"] = horizon_hours
    audit["observed_future_hours"] = 0
    audit["missing_future_hours"] = horizon_hours
    audit["positive_future_hours"] = 0
    audit["target_status"] = "unknown"

    for row_i, t in enumerate(prediction_index):
        future_times = pd.date_range(
            t + pd.Timedelta(hours=1),
            periods=horizon_hours,
            freq="h",
        )

        future_values = hourly.reindex(future_times)

        observed_mask = future_values.notna()
        positive_mask = (
            observed_mask
            & (future_values >= threshold)
        )

        observed_count = int(observed_mask.sum())
        positive_count = int(positive_mask.sum())
        missing_count = horizon_hours - observed_count

        audit.iat[
            row_i,
            audit.columns.get_loc("observed_future_hours"),
        ] = observed_count
        audit.iat[
            row_i,
            audit.columns.get_loc("missing_future_hours"),
        ] = missing_count
        audit.iat[
            row_i,
            audit.columns.get_loc("positive_future_hours"),
        ] = positive_count

        if positive_count > 0:
            target.iat[row_i] = 1.0
            audit.iat[
                row_i,
                audit.columns.get_loc("target_status"),
            ] = "positive"

        elif missing_count == 0:
            target.iat[row_i] = 0.0
            audit.iat[
                row_i,
                audit.columns.get_loc("target_status"),
            ] = "negative"

        else:
            # Keep NaN: the existential condition was not observed,
            # but the complete future horizon is not known.
            audit.iat[
                row_i,
                audit.columns.get_loc("target_status"),
            ] = "unknown"

    if return_audit:
        return target, audit

    return target
