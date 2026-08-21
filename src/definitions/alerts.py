"""Canonical operational alert construction and evaluation.

This module converts hourly model probabilities into alert episodes and
associates those episodes with canonical geomagnetic-storm events.

Frozen primary semantics
------------------------
H = 6 hours
C = 3 hours
maximum FAR/day = 0.2

An hourly alert is generated when:

    probability >= threshold

Alert timestamps separated by at most C hours belong to the same episode.

Missing probabilities are unknown forecasts. They:

- do not generate alerts;
- do not count as negative predictions;
- reduce valid prediction exposure;
- do not independently terminate an episode.

Episode grouping is determined by elapsed wall-clock time between actual
alert timestamps.

Each alert episode may be associated with at most one canonical storm event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_HORIZON_HOURS = 6
DEFAULT_ALERT_COOLDOWN_HOURS = 3


_EPISODE_COLUMNS = [
    "alert_episode_id",
    "first_alert_time",
    "last_alert_time",
    "classification",
    "associated_event_id",
    "lead_time",
    "n_alert_hours",
    "max_probability",
    "threshold",
]


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )


def _validate_threshold(
    threshold: float,
) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "threshold must be numeric."
        ) from exc

    if not np.isfinite(value):
        raise ValueError(
            "threshold must be finite."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    return value


def _validate_probabilities(
    probabilities: pd.Series,
) -> pd.Series:
    """Validate an hourly probability series.

    Missing probability values are permitted.

    Missing timestamps, duplicate timestamps, non-hourly alignment, and
    non-monotonic ordering are not permitted.
    """

    if not isinstance(
        probabilities,
        pd.Series,
    ):
        raise TypeError(
            "probabilities must be a pandas Series."
        )

    if not isinstance(
        probabilities.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "probabilities.index must be a pandas DatetimeIndex."
        )

    if probabilities.index.hasnans:
        raise ValueError(
            "probability timestamps must not contain NaT."
        )

    if probabilities.index.has_duplicates:
        raise ValueError(
            "probability timestamps must not contain duplicates."
        )

    if not probabilities.index.is_monotonic_increasing:
        raise ValueError(
            "probability timestamps must be monotonically increasing."
        )

    if (
        (
            probabilities.index.minute != 0
        ).any()
        or (
            probabilities.index.second != 0
        ).any()
        or (
            probabilities.index.microsecond != 0
        ).any()
    ):
        raise ValueError(
            "probability timestamps must be aligned to whole hours."
        )

    try:
        numeric = pd.to_numeric(
            probabilities,
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "probabilities must contain only numeric values or NaN."
        ) from exc

    finite = numeric.notna()

    invalid = (
        finite
        & (
            (numeric < 0.0)
            | (numeric > 1.0)
            | ~np.isfinite(numeric)
        )
    )

    if invalid.any():
        raise ValueError(
            "finite probabilities must lie between 0 and 1."
        )

    return numeric


def _empty_episodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alert_episode_id": pd.Series(
                dtype="int64"
            ),
            "first_alert_time": pd.Series(
                dtype="datetime64[ns]"
            ),
            "last_alert_time": pd.Series(
                dtype="datetime64[ns]"
            ),
            "classification": pd.Series(
                dtype="object"
            ),
            "associated_event_id": pd.Series(
                dtype="Int64"
            ),
            "lead_time": pd.Series(
                dtype="timedelta64[ns]"
            ),
            "n_alert_hours": pd.Series(
                dtype="int64"
            ),
            "max_probability": pd.Series(
                dtype="float64"
            ),
            "threshold": pd.Series(
                dtype="float64"
            ),
        }
    )


def identify_alerts(
    probabilities: pd.Series,
    threshold: float,
    *,
    cooldown_hours: int = (
        DEFAULT_ALERT_COOLDOWN_HOURS
    ),
) -> pd.DataFrame:
    """Convert hourly probabilities into operational alert episodes.

    Episodes are grouped using elapsed wall-clock time between alert
    timestamps:

        gap <= C  -> same episode
        gap > C   -> new episode

    Missing probabilities do not generate alerts. They do not by themselves
    terminate an episode.
    """

    threshold = _validate_threshold(
        threshold
    )

    _validate_positive_integer(
        cooldown_hours,
        name="cooldown_hours",
    )

    probabilities = _validate_probabilities(
        probabilities
    )

    valid = probabilities.notna()

    alert_mask = (
        valid
        & (
            probabilities >= threshold
        )
    )

    alert_times = probabilities.index[
        alert_mask
    ]

    if len(alert_times) == 0:
        return _empty_episodes()

    cooldown = pd.Timedelta(
        hours=cooldown_hours
    )

    episodes: list[dict[str, object]] = []

    current_times: list[pd.Timestamp] = []

    for timestamp in alert_times:
        if not current_times:
            current_times = [
                timestamp
            ]
            continue

        gap = (
            timestamp
            - current_times[-1]
        )

        if gap <= cooldown:
            current_times.append(
                timestamp
            )
            continue

        values = probabilities.loc[
            current_times
        ]

        episodes.append(
            {
                "alert_episode_id": (
                    len(episodes) + 1
                ),
                "first_alert_time": (
                    current_times[0]
                ),
                "last_alert_time": (
                    current_times[-1]
                ),
                "classification": None,
                "associated_event_id": pd.NA,
                "lead_time": pd.NaT,
                "n_alert_hours": len(
                    current_times
                ),
                "max_probability": float(
                    values.max()
                ),
                "threshold": threshold,
            }
        )

        current_times = [
            timestamp
        ]

    values = probabilities.loc[
        current_times
    ]

    episodes.append(
        {
            "alert_episode_id": (
                len(episodes) + 1
            ),
            "first_alert_time": (
                current_times[0]
            ),
            "last_alert_time": (
                current_times[-1]
            ),
            "classification": None,
            "associated_event_id": pd.NA,
            "lead_time": pd.NaT,
            "n_alert_hours": len(
                current_times
            ),
            "max_probability": float(
                values.max()
            ),
            "threshold": threshold,
        }
    )

    result = pd.DataFrame(
        episodes,
        columns=_EPISODE_COLUMNS,
    )

    result[
        "associated_event_id"
    ] = result[
        "associated_event_id"
    ].astype(
        "Int64"
    )

    return result


def _validate_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "event_id",
        "start_time",
        "end_time",
        "boundary_status",
    }

    if not isinstance(
        events,
        pd.DataFrame,
    ):
        raise TypeError(
            "events must be a pandas DataFrame."
        )

    missing = (
        required
        - set(events.columns)
    )

    if missing:
        raise ValueError(
            "events is missing required columns: "
            f"{sorted(missing)}"
        )

    frame = events.copy()

    frame["start_time"] = pd.to_datetime(
        frame["start_time"],
        errors="raise",
    )

    frame["end_time"] = pd.to_datetime(
        frame["end_time"],
        errors="raise",
    )

    if frame[
        "start_time"
    ].isna().any():
        raise ValueError(
            "event start_time must not contain NaT."
        )

    if frame[
        "event_id"
    ].duplicated().any():
        raise ValueError(
            "event_id must be unique."
        )

    if not frame[
        "start_time"
    ].is_monotonic_increasing:
        raise ValueError(
            "events must be ordered by start_time."
        )

    finite_end = frame[
        "end_time"
    ].notna()

    if (
        frame.loc[
            finite_end,
            "end_time",
        ]
        < frame.loc[
            finite_end,
            "start_time",
        ]
    ).any():
        raise ValueError(
            "event end_time cannot precede start_time."
        )

    allowed_status = {
        "complete",
        "left_censored",
        "right_censored",
        "both_censored",
    }

    if not frame[
        "boundary_status"
    ].isin(
        allowed_status
    ).all():
        raise ValueError(
            "events contains an invalid boundary_status."
        )

    return frame.reset_index(
        drop=True
    )


def associate_alerts_with_events(
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_hours: int = (
        DEFAULT_HORIZON_HOURS
    ),
) -> pd.DataFrame:
    """Associate alert episodes with canonical storm events.

    Association uses each episode's ``first_alert_time``.

    For each episode, events are considered chronologically.

    Early Detection:

        storm_start - H <= first_alert_time < storm_start

    Late Detection:

        storm_start <= first_alert_time <= storm_end

    Right-censored events have no finite canonical end time. Their Early
    Detection window remains usable, but this function does not invent an
    unbounded Late Detection interval.

    Each episode is associated with at most one event.
    """

    _validate_positive_integer(
        horizon_hours,
        name="horizon_hours",
    )

    if not isinstance(
        episodes,
        pd.DataFrame,
    ):
        raise TypeError(
            "episodes must be a pandas DataFrame."
        )

    required_episode_columns = {
        "alert_episode_id",
        "first_alert_time",
        "last_alert_time",
        "classification",
        "associated_event_id",
        "lead_time",
    }

    missing = (
        required_episode_columns
        - set(episodes.columns)
    )

    if missing:
        raise ValueError(
            "episodes is missing required columns: "
            f"{sorted(missing)}"
        )

    result = episodes.copy()

    if result.empty:
        return result

    result[
        "first_alert_time"
    ] = pd.to_datetime(
        result["first_alert_time"],
        errors="raise",
    )

    result[
        "last_alert_time"
    ] = pd.to_datetime(
        result["last_alert_time"],
        errors="raise",
    )

    if result[
        "first_alert_time"
    ].isna().any():
        raise ValueError(
            "first_alert_time must not contain NaT."
        )

    if result[
        "alert_episode_id"
    ].duplicated().any():
        raise ValueError(
            "alert_episode_id must be unique."
        )

    if not result[
        "first_alert_time"
    ].is_monotonic_increasing:
        raise ValueError(
            "episodes must be ordered by first_alert_time."
        )

    event_frame = _validate_events(
        events
    )

    horizon = pd.Timedelta(
        hours=horizon_hours
    )

    classifications: list[str] = []
    associated_ids: list[object] = []
    lead_times: list[object] = []

    for episode in result.itertuples():
        alert_time = episode.first_alert_time

        classification = "false_alarm"
        associated_event_id: object = pd.NA
        lead_time: object = pd.NaT

        # Early association is checked first.
        for event in event_frame.itertuples(
            index=False
        ):
            early_start = (
                event.start_time
                - horizon
            )

            if (
                early_start
                <= alert_time
                < event.start_time
            ):
                classification = (
                    "early_detection"
                )
                associated_event_id = (
                    event.event_id
                )
                lead_time = (
                    event.start_time
                    - alert_time
                )
                break

        # Only if no Early Detection exists do we test Late Detection.
        if classification == "false_alarm":
            for event in event_frame.itertuples(
                index=False
            ):
                if pd.isna(
                    event.end_time
                ):
                    continue

                if (
                    event.start_time
                    <= alert_time
                    <= event.end_time
                ):
                    classification = (
                        "late_detection"
                    )
                    associated_event_id = (
                        event.event_id
                    )
                    lead_time = pd.NaT
                    break

        classifications.append(
            classification
        )
        associated_ids.append(
            associated_event_id
        )
        lead_times.append(
            lead_time
        )

    result["classification"] = (
        classifications
    )

    result[
        "associated_event_id"
    ] = pd.array(
        associated_ids,
        dtype="Int64",
    )

    result["lead_time"] = pd.to_timedelta(
        lead_times
    )

    return result


def event_recall(
    episodes: pd.DataFrame,
    events: pd.DataFrame,
) -> float:
    """Compute canonical event-level recall.

    Additional alert episodes associated with an already detected event do
    not increase recall.
    """

    event_frame = _validate_events(
        events
    )

    if event_frame.empty:
        return np.nan

    if "classification" not in episodes.columns:
        raise ValueError(
            "episodes must contain classification."
        )

    if "associated_event_id" not in episodes.columns:
        raise ValueError(
            "episodes must contain associated_event_id."
        )

    detections = episodes[
        episodes["classification"].isin(
            [
                "early_detection",
                "late_detection",
            ]
        )
    ]

    detected_ids = set(
        detections[
            "associated_event_id"
        ].dropna().tolist()
    )

    valid_event_ids = set(
        event_frame[
            "event_id"
        ].tolist()
    )

    detected_ids &= valid_event_ids

    return (
        len(detected_ids)
        / len(valid_event_ids)
    )


def valid_prediction_exposure_days(
    probabilities: pd.Series,
) -> float:
    """Return valid prediction exposure in 24-hour days."""

    probabilities = _validate_probabilities(
        probabilities
    )

    valid_hours = int(
        probabilities.notna().sum()
    )

    return (
        valid_hours
        / 24.0
    )


def false_alarm_rate_per_day(
    episodes: pd.DataFrame,
    probabilities: pd.Series,
) -> float:
    """Compute false alert episodes per valid prediction exposure day."""

    exposure_days = (
        valid_prediction_exposure_days(
            probabilities
        )
    )

    if exposure_days == 0:
        return np.nan

    if "classification" not in episodes.columns:
        raise ValueError(
            "episodes must contain classification."
        )

    false_alarms = int(
        (
            episodes["classification"]
            == "false_alarm"
        ).sum()
    )

    return (
        false_alarms
        / exposure_days
    )


def early_detection_lead_times(
    episodes: pd.DataFrame,
) -> pd.Series:
    """Return one lead time per early-detected event.

    When multiple Early Detection episodes are associated with the same
    event, the earliest qualifying episode is used, which corresponds to
    the maximum valid lead time for that event.
    """

    required = {
        "first_alert_time",
        "classification",
        "associated_event_id",
        "lead_time",
    }

    missing = (
        required
        - set(episodes.columns)
    )

    if missing:
        raise ValueError(
            "episodes is missing required columns: "
            f"{sorted(missing)}"
        )

    early = episodes[
        episodes["classification"]
        == "early_detection"
    ].copy()

    if early.empty:
        return pd.Series(
            dtype="timedelta64[ns]",
            name="lead_time",
        )

    early = early.sort_values(
        [
            "associated_event_id",
            "first_alert_time",
        ]
    )

    earliest = early.drop_duplicates(
        subset=[
            "associated_event_id"
        ],
        keep="first",
    )

    result = pd.to_timedelta(
        earliest["lead_time"]
    ).reset_index(
        drop=True
    )

    result.name = "lead_time"

    return result