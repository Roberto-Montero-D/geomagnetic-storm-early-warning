"""Canonical geomagnetic-storm event construction.

Storm events are constructed from retrospective ground-truth Kp, not from
predictor-side causal Kp features.

Frozen primary semantics
------------------------
T = 5
Z = 6 hours

A storm begins at the start of the first canonical Kp interval satisfying
Kp >= T.

For segmentation, canonical 3-hour Kp intervals are expanded onto an hourly
ground-truth timeline. An active event terminates only after Z consecutive
valid hourly states with Kp < T.

Missing Kp intervals do not count as quiet time and break a candidate
below-threshold termination run.

Dataset-boundary events are explicitly censored rather than assigned
invented physical boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_THRESHOLD = 5.0
DEFAULT_TERMINATION_HOURS = 6

_REQUIRED_COLUMNS = {
    "interval_start",
    "interval_end",
    "kp",
}


@dataclass(frozen=True)
class _HourlyState:
    """One retrospective hourly ground-truth state."""

    timestamp: pd.Timestamp
    kp: float


def _validate_parameters(
    threshold: float,
    termination_hours: int,
) -> None:
    """Validate event-definition parameters."""

    if not np.isfinite(threshold):
        raise ValueError(
            "threshold must be finite."
        )

    if (
        not isinstance(termination_hours, int)
        or isinstance(termination_hours, bool)
    ):
        raise TypeError(
            "termination_hours must be an integer."
        )

    if termination_hours <= 0:
        raise ValueError(
            "termination_hours must be greater than zero."
        )


def _validate_intervals(
    kp_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """Validate canonical Kp interval structure."""

    if not isinstance(kp_intervals, pd.DataFrame):
        raise TypeError(
            "kp_intervals must be a pandas DataFrame."
        )

    missing_columns = (
        _REQUIRED_COLUMNS
        - set(kp_intervals.columns)
    )

    if missing_columns:
        raise ValueError(
            "kp_intervals is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame = kp_intervals[
        [
            "interval_start",
            "interval_end",
            "kp",
        ]
    ].copy()

    frame["interval_start"] = pd.to_datetime(
        frame["interval_start"],
        errors="coerce",
    )
    frame["interval_end"] = pd.to_datetime(
        frame["interval_end"],
        errors="coerce",
    )

    if frame["interval_start"].isna().any():
        raise ValueError(
            "interval_start must not contain NaT."
        )

    if frame["interval_end"].isna().any():
        raise ValueError(
            "interval_end must not contain NaT."
        )

    if frame["interval_start"].duplicated().any():
        raise ValueError(
            "interval_start contains duplicate timestamps."
        )

    if not frame[
        "interval_start"
    ].is_monotonic_increasing:
        raise ValueError(
            "Kp intervals must be ordered by interval_start."
        )

    durations = (
        frame["interval_end"]
        - frame["interval_start"]
    )

    expected_duration = pd.Timedelta(
        hours=3
    )

    if not durations.eq(
        expected_duration
    ).all():
        raise ValueError(
            "Every canonical Kp interval must span exactly 3 hours."
        )

    if len(frame) > 1:
        starts = frame[
            "interval_start"
        ].iloc[1:].reset_index(
            drop=True
        )

        previous_ends = frame[
            "interval_end"
        ].iloc[:-1].reset_index(
            drop=True
        )

        overlaps = (
            starts
            < previous_ends
        )

        if overlaps.any():
            raise ValueError(
                "Canonical Kp intervals must not overlap."
            )

    frame["kp"] = pd.to_numeric(
        frame["kp"],
        errors="raise",
    )

    finite_or_missing = (
        frame["kp"].isna()
        | np.isfinite(
            frame["kp"].to_numpy(
                dtype=float
            )
        )
    )

    if not bool(
        np.all(finite_or_missing)
    ):
        raise ValueError(
            "kp must contain only finite values or NaN."
        )

    return frame.reset_index(
        drop=True
    )


def _expand_to_hourly(
    kp_intervals: pd.DataFrame,
) -> list[_HourlyState]:
    """Expand canonical 3-hour Kp intervals to hourly ground truth.

    The implementation is vectorized. Temporal gaps between canonical
    intervals remain explicit NaN hourly states, preserving the frozen event
    semantics that missing data cannot count as quiet time.
    """

    if kp_intervals.empty:
        return []

    starts = pd.DatetimeIndex(
        kp_intervals["interval_start"]
    )
    ends = pd.DatetimeIndex(
        kp_intervals["interval_end"]
    )
    kp_values = kp_intervals["kp"].to_numpy(
        dtype=float,
        copy=False,
    )

    first_start = starts[0]
    final_end = ends[-1]

    hourly_index = pd.date_range(
        start=first_start,
        end=final_end,
        freq="h",
        inclusive="left",
    )

    hourly_values = np.full(
        len(hourly_index),
        np.nan,
        dtype=float,
    )

    # Each validated canonical interval spans exactly three hours. Convert
    # interval starts directly into integer offsets from the first hour and
    # assign all three positions without constructing one DatetimeIndex per
    # interval or performing repeated .loc alignment.
    # Compute offsets as timedeltas rather than mixing ``DatetimeIndex.asi8``
    # with ``Timestamp.value``. Pandas may store the index in microseconds
    # while Timestamp.value is nanoseconds.
    start_offsets = np.asarray(
        (starts - first_start) / pd.Timedelta(hours=1),
        dtype=np.int64,
    )

    positions = (
        start_offsets[:, None]
        + np.arange(
            3,
            dtype=np.int64,
        )[None, :]
    ).reshape(-1)

    repeated_kp = np.repeat(
        kp_values,
        3,
    )

    hourly_values[positions] = repeated_kp

    # Keep the same external representation as the original implementation so
    # identify_events and its tests retain identical behavior.
    return [
        _HourlyState(
            timestamp=pd.Timestamp(timestamp),
            kp=(
                float(value)
                if not np.isnan(value)
                else np.nan
            ),
        )
        for timestamp, value in zip(
            hourly_index,
            hourly_values,
            strict=True,
        )
    ]


def _empty_events() -> pd.DataFrame:
    """Return an empty event table with stable columns."""

    return pd.DataFrame(
        {
            "event_id": pd.Series(
                dtype="int64"
            ),
            "start_time": pd.Series(
                dtype="datetime64[ns]"
            ),
            "end_time": pd.Series(
                dtype="datetime64[ns]"
            ),
            "threshold": pd.Series(
                dtype="float64"
            ),
            "peak_kp": pd.Series(
                dtype="float64"
            ),
            "boundary_status": pd.Series(
                dtype="object"
            ),
        }
    )


def _boundary_status(
    *,
    left_censored: bool,
    right_censored: bool,
) -> str:
    """Return canonical boundary-status label."""

    if (
        left_censored
        and right_censored
    ):
        return "both_censored"

    if left_censored:
        return "left_censored"

    if right_censored:
        return "right_censored"

    return "complete"


def identify_events(
    kp_intervals: pd.DataFrame,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    termination_hours: int = (
        DEFAULT_TERMINATION_HOURS
    ),
) -> pd.DataFrame:
    """Identify canonical geomagnetic-storm events."""

    _validate_parameters(
        threshold,
        termination_hours,
    )

    intervals = _validate_intervals(
        kp_intervals
    )

    if intervals.empty:
        return _empty_events()

    hourly = _expand_to_hourly(
        intervals
    )

    if not hourly:
        return _empty_events()

    events: list[dict[str, object]] = []

    active = False
    event_start: pd.Timestamp | None = None
    event_peak = np.nan
    left_censored = False

    quiet_run = 0
    quiet_run_start: pd.Timestamp | None = None

    first_timestamp = hourly[0].timestamp

    for state in hourly:
        timestamp = state.timestamp
        kp = state.kp

        if not active:
            if np.isnan(kp):
                continue

            if kp >= threshold:
                active = True
                event_start = timestamp
                event_peak = kp
                left_censored = (
                    timestamp
                    == first_timestamp
                )

                quiet_run = 0
                quiet_run_start = None

            continue

        if np.isnan(kp):
            quiet_run = 0
            quiet_run_start = None
            continue

        if kp >= threshold:
            event_peak = max(
                event_peak,
                kp,
            )

            quiet_run = 0
            quiet_run_start = None
            continue

        if quiet_run == 0:
            quiet_run_start = timestamp

        quiet_run += 1

        if quiet_run < termination_hours:
            continue

        assert event_start is not None
        assert quiet_run_start is not None

        end_time = (
            quiet_run_start
            - pd.Timedelta(
                hours=1
            )
        )

        events.append(
            {
                "event_id": (
                    len(events) + 1
                ),
                "start_time": event_start,
                "end_time": end_time,
                "threshold": float(
                    threshold
                ),
                "peak_kp": float(
                    event_peak
                ),
                "boundary_status": (
                    _boundary_status(
                        left_censored=left_censored,
                        right_censored=False,
                    )
                ),
            }
        )

        active = False
        event_start = None
        event_peak = np.nan
        left_censored = False
        quiet_run = 0
        quiet_run_start = None

    if active:
        assert event_start is not None

        events.append(
            {
                "event_id": (
                    len(events) + 1
                ),
                "start_time": event_start,
                "end_time": pd.NaT,
                "threshold": float(
                    threshold
                ),
                "peak_kp": float(
                    event_peak
                ),
                "boundary_status": (
                    _boundary_status(
                        left_censored=left_censored,
                        right_censored=True,
                    )
                ),
            }
        )

    if not events:
        return _empty_events()

    result = pd.DataFrame(
        events
    )

    result["event_id"] = result[
        "event_id"
    ].astype(
        "int64"
    )

    result["start_time"] = pd.to_datetime(
        result["start_time"]
    )

    result["end_time"] = pd.to_datetime(
        result["end_time"]
    )

    result["threshold"] = result[
        "threshold"
    ].astype(
        "float64"
    )

    result["peak_kp"] = result[
        "peak_kp"
    ].astype(
        "float64"
    )

    return result
