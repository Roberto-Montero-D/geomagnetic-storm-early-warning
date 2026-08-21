"""Generic temporal cutoff utilities.

This module implements the project's canonical predictor-side
information cutoff.

For prediction time ``t``:

    information_cutoff = t - 1h

For an interval-based observation beginning at ``s`` with duration ``d``:

    period_start = s
    period_end   = s + d

The observation is causally eligible only when:

    period_end <= information_cutoff

For the project's hourly OMNI measurements:

    d = 1h

therefore:

    s + 1h <= t - 1h

This module does not construct features, targets, Kp intervals, or
perform source-specific missing-value handling.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEFAULT_INFORMATION_DELAY = pd.Timedelta(
    hours=1
)

DEFAULT_INTERVAL_DURATION = pd.Timedelta(
    hours=1
)


def _as_timestamp(
    value: pd.Timestamp | str,
    *,
    name: str,
) -> pd.Timestamp:
    """Normalize and validate a scalar timestamp."""

    timestamp = pd.Timestamp(value)

    if pd.isna(timestamp):
        raise ValueError(
            f"{name} must not be NaT."
        )

    return timestamp


def _validate_datetime_index(
    index: pd.DatetimeIndex,
    *,
    name: str,
) -> None:
    """Validate timestamp integrity without requiring continuity.

    Missing timestamps are permitted because a source can legitimately
    contain gaps. They must remain gaps rather than being silently
    invented or interpreted as quiet measurements.
    """

    if not isinstance(
        index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            f"{name} must be a pandas DatetimeIndex."
        )

    if index.hasnans:
        raise ValueError(
            f"{name} must not contain NaT."
        )

    if index.has_duplicates:
        duplicates = (
            index[index.duplicated()]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{name} contains duplicate timestamps: "
            f"{duplicates}"
        )

    if not index.is_monotonic_increasing:
        raise ValueError(
            f"{name} must be monotonically increasing."
        )


def _validate_positive_timedelta(
    value: pd.Timedelta,
    *,
    name: str,
) -> None:
    """Require a finite positive duration."""

    if pd.isna(value):
        raise ValueError(
            f"{name} must not be NaT."
        )

    if value <= pd.Timedelta(0):
        raise ValueError(
            f"{name} must be greater than zero."
        )


def information_cutoff(
    prediction_time: pd.Timestamp | str,
    *,
    information_delay: pd.Timedelta = (
        DEFAULT_INFORMATION_DELAY
    ),
) -> pd.Timestamp:
    """Return the latest information time allowed for prediction.

    Canonical project rule:

        cutoff(t) = t - 1h
    """

    prediction_time = _as_timestamp(
        prediction_time,
        name="prediction_time",
    )

    information_delay = pd.Timedelta(
        information_delay
    )

    _validate_positive_timedelta(
        information_delay,
        name="information_delay",
    )

    return (
        prediction_time
        - information_delay
    )


def interval_end_times(
    observation_times: (
        pd.DatetimeIndex
        | Iterable[pd.Timestamp]
    ),
    *,
    interval_duration: pd.Timedelta = (
        DEFAULT_INTERVAL_DURATION
    ),
) -> pd.DatetimeIndex:
    """Convert interval-start timestamps to interval-end timestamps."""

    index = pd.DatetimeIndex(
        observation_times
    )

    _validate_datetime_index(
        index,
        name="observation_times",
    )

    interval_duration = pd.Timedelta(
        interval_duration
    )

    _validate_positive_timedelta(
        interval_duration,
        name="interval_duration",
    )

    return pd.DatetimeIndex(
        index + interval_duration,
        name="period_end",
    )


def eligible_interval_mask(
    observation_times: (
        pd.DatetimeIndex
        | Iterable[pd.Timestamp]
    ),
    prediction_time: pd.Timestamp | str,
    *,
    interval_duration: pd.Timedelta = (
        DEFAULT_INTERVAL_DURATION
    ),
    information_delay: pd.Timedelta = (
        DEFAULT_INFORMATION_DELAY
    ),
) -> pd.Series:
    """Return causal eligibility for interval-based observations.

    An observation beginning at ``s`` is eligible only when:

        s + interval_duration
            <=
        prediction_time - information_delay

    The returned Series is indexed by the original observation times.
    """

    observation_index = pd.DatetimeIndex(
        observation_times
    )

    _validate_datetime_index(
        observation_index,
        name="observation_times",
    )

    cutoff = information_cutoff(
        prediction_time,
        information_delay=information_delay,
    )

    ends = interval_end_times(
        observation_index,
        interval_duration=interval_duration,
    )

    return pd.Series(
        ends <= cutoff,
        index=observation_index,
        name="eligible",
        dtype=bool,
    )


def select_eligible_intervals(
    frame: pd.DataFrame,
    prediction_time: pd.Timestamp | str,
    *,
    interval_duration: pd.Timedelta = (
        DEFAULT_INTERVAL_DURATION
    ),
    information_delay: pd.Timedelta = (
        DEFAULT_INFORMATION_DELAY
    ),
) -> pd.DataFrame:
    """Return only observations causally available at prediction time.

    The input frame is never modified.

    Source gaps remain gaps. No forward fill, backward fill, interpolation,
    or timeline reconstruction is performed.
    """

    if not isinstance(
        frame.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "frame.index must be a pandas DatetimeIndex."
        )

    _validate_datetime_index(
        frame.index,
        name="frame.index",
    )

    mask = eligible_interval_mask(
        frame.index,
        prediction_time,
        interval_duration=interval_duration,
        information_delay=information_delay,
    )

    return frame.loc[
        mask.to_numpy()
    ].copy()


def maximum_eligible_information_time(
    frame: pd.DataFrame,
    prediction_time: pd.Timestamp | str,
    *,
    interval_duration: pd.Timedelta = (
        DEFAULT_INTERVAL_DURATION
    ),
    information_delay: pd.Timedelta = (
        DEFAULT_INFORMATION_DELAY
    ),
) -> pd.Timestamp | None:
    """Return the maximum period-end time visible to a prediction.

    This helper is intended primarily for causality audits.

    Returns ``None`` when no observation is eligible.
    """

    eligible = select_eligible_intervals(
        frame,
        prediction_time,
        interval_duration=interval_duration,
        information_delay=information_delay,
    )

    if eligible.empty:
        return None

    latest_start = eligible.index[-1]

    return (
        latest_start
        + pd.Timedelta(
            interval_duration
        )
    )