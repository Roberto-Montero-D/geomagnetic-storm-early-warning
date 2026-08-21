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

    Gaps between canonical intervals are explicitly represented as missing
    hourly states. This is essential because a missing interval must not be
    interpreted as quiet time.
    """

    if kp_intervals.empty:
        return []

    first_start = kp_intervals[
        "interval_start"
    ].iloc[0]

    final_end = kp_intervals[
        "interval_end"
    ].iloc[-1]

    hourly_index = pd.date_range(
        start=first_start,
        end=final_end,
        freq="h",
        inclusive="left",
    )

    hourly = pd.Series(
        np.nan,
        index=hourly_index,
        dtype=float,
    )

    for row in kp_intervals.itertuples(
        index=False
    ):
        interval_hours = pd.date_range(
            start=row.interval_start,
            end=row.interval_end,
            freq="h",
            inclusive="left",
        )

        hourly.loc[
            interval_hours
        ] = row.kp

    return [
        _HourlyState(
            timestamp=timestamp,
            kp=float(value)
            if pd.notna(value)
            else np.nan,
        )
        for timestamp, value
        in hourly.items()
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
    """Identify canonical geomagnetic-storm events.

    Parameters
    ----------
    kp_intervals
        Canonical Kp interval table containing:

            interval_start
            interval_end
            kp

        Intervals must be ordered, non-overlapping, and exactly three
        hours long. Missing canonical intervals may appear as temporal
        gaps. Explicit missing Kp values are also permitted.

    threshold
        Storm threshold. Primary protocol value: 5.

    termination_hours
        Number of consecutive valid hourly states below threshold required
        to terminate an active event. Primary protocol value: 6.

    Returns
    -------
    pandas.DataFrame
        Columns:

            event_id
            start_time
            end_time
            threshold
            peak_kp
            boundary_status

        ``end_time`` is the final active hourly timestamp before the
        terminating quiet run.

        For a right-censored event, ``end_time`` is NaT because true
        termination was not observed.

    Notes
    -----
    Missing Kp states:

    - do not start an event;
    - do not terminate an event;
    - do not count toward the quiet run;
    - reset an in-progress quiet run.

    The first observed storm state is marked left-censored only when the
    dataset begins in storm conditions. This prevents the first available
    timestamp from being asserted as the physical storm onset.
    """

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

        # Active event.
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

        # Valid below-threshold state.
        if quiet_run == 0:
            quiet_run_start = timestamp

        quiet_run += 1

        if quiet_run < termination_hours:
            continue

        assert event_start is not None
        assert quiet_run_start is not None

        # The quiet run itself is not part of the storm.
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