"""Canonical hourly prediction-time grid.

Phase 1.1 defines only the universe of prediction timestamps. It does not
inspect OMNI/Kp row availability, build features or targets, assign temporal
splits, or drop rows.

The grid is therefore a calendar object, not a data-availability object.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_GRID_START = pd.Timestamp("1996-01-01 00:00:00")
DEFAULT_GRID_END_EXCLUSIVE = pd.Timestamp("2026-01-01 00:00:00")
PREDICTION_TIME_NAME = "prediction_time"


def _as_naive_hour(
    value: object,
    *,
    name: str,
) -> pd.Timestamp:
    """Validate and normalize one grid boundary."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be convertible to pandas.Timestamp."
        ) from exc

    if pd.isna(timestamp):
        raise ValueError(f"{name} must not be NaT.")

    if timestamp.tz is not None:
        raise ValueError(
            f"{name} must be timezone-naive; the canonical project "
            "timeline uses naive UTC-like timestamps."
        )

    if (
        timestamp.minute != 0
        or timestamp.second != 0
        or timestamp.microsecond != 0
        or timestamp.nanosecond != 0
    ):
        raise ValueError(
            f"{name} must be aligned to an exact whole hour."
        )

    return timestamp


def build_prediction_grid(
    *,
    start: object = DEFAULT_GRID_START,
    end_exclusive: object = DEFAULT_GRID_END_EXCLUSIVE,
) -> pd.DatetimeIndex:
    """Return the canonical continuous hourly prediction grid.

    The interval is half-open:

        [start, end_exclusive)

    Defaults correspond to the frozen primary study coverage:

        1996-01-01 00:00 <= prediction_time < 2026-01-01 00:00

    Thus the final default timestamp is:

        2025-12-31 23:00

    This function intentionally does not inspect source data. Missing OMNI,
    Kp, features, or targets must remain properties of later dataset rows
    rather than causing prediction timestamps to disappear.
    """

    start_ts = _as_naive_hour(start, name="start")
    end_ts = _as_naive_hour(
        end_exclusive,
        name="end_exclusive",
    )

    if start_ts >= end_ts:
        raise ValueError(
            "start must be strictly earlier than end_exclusive."
        )

    grid = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq="h",
        inclusive="left",
        name=PREDICTION_TIME_NAME,
    )

    # Defensive invariants. These should follow from date_range, but keeping
    # them explicit makes the canonical grid contract auditable.
    if grid.hasnans:
        raise RuntimeError(
            "Canonical prediction grid unexpectedly contains NaT."
        )

    if grid.has_duplicates:
        raise RuntimeError(
            "Canonical prediction grid unexpectedly contains duplicates."
        )

    if not grid.is_monotonic_increasing:
        raise RuntimeError(
            "Canonical prediction grid is not monotonically increasing."
        )

    if len(grid) > 1:
        deltas = grid[1:] - grid[:-1]
        if not (deltas == pd.Timedelta(hours=1)).all():
            raise RuntimeError(
                "Canonical prediction grid is not exactly hourly."
            )

    return grid
