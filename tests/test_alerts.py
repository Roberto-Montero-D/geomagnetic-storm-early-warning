import numpy as np
import pandas as pd
import pytest

from src.definitions.alerts import (
    associate_alerts_with_events,
    early_detection_lead_times,
    event_recall,
    false_alarm_rate_per_day,
    identify_alerts,
    valid_prediction_exposure_days,
)


def _probabilities(
    values,
    *,
    start="2020-01-01 00:00",
):
    return pd.Series(
        values,
        index=pd.date_range(
            start,
            periods=len(values),
            freq="h",
        ),
        dtype=float,
    )


def _events(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "start_time",
            "end_time",
            "boundary_status",
        ],
    )


def _associate(
    probabilities,
    events,
    *,
    threshold=0.5,
):
    episodes = identify_alerts(
        probabilities,
        threshold,
    )

    return associate_alerts_with_events(
        episodes,
        events,
    )


def test_single_alert_episode():
    probabilities = _probabilities(
        [0.1, 0.8, 0.1]
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
    )

    assert len(episodes) == 1
    assert episodes.loc[
        0, "first_alert_time"
    ] == pd.Timestamp(
        "2020-01-01 01:00"
    )
    assert episodes.loc[
        0, "last_alert_time"
    ] == pd.Timestamp(
        "2020-01-01 01:00"
    )
    assert episodes.loc[
        0, "n_alert_hours"
    ] == 1


def test_consecutive_alerts_same_episode():
    probabilities = _probabilities(
        [0.8, 0.9, 0.7, 0.1]
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
    )

    assert len(episodes) == 1
    assert episodes.loc[
        0, "n_alert_hours"
    ] == 3


def test_alert_gap_equal_to_c():
    probabilities = _probabilities(
        [0.8, 0.1, 0.1, 0.8]
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
        cooldown_hours=3,
    )

    assert len(episodes) == 1


def test_alert_gap_greater_than_c():
    probabilities = _probabilities(
        [0.8, 0.1, 0.1, 0.1, 0.8]
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
        cooldown_hours=3,
    )

    assert len(episodes) == 2


def test_missing_probability_does_not_generate_alert():
    probabilities = _probabilities(
        [0.8, np.nan, 0.1]
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
    )

    assert len(episodes) == 1
    assert episodes.loc[
        0, "n_alert_hours"
    ] == 1


def test_missing_probability_does_not_break_episode_by_itself():
    probabilities = _probabilities(
        [0.8, np.nan, 0.8]
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
        cooldown_hours=3,
    )

    assert len(episodes) == 1
    assert episodes.loc[
        0, "n_alert_hours"
    ] == 2


def test_actual_timestamp_gap_controls_episode_grouping():
    probabilities = pd.Series(
        [0.8, 0.8],
        index=pd.DatetimeIndex(
            [
                "2020-01-01 10:00",
                "2020-01-01 14:00",
            ]
        ),
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
        cooldown_hours=3,
    )

    assert len(episodes) == 2


def test_threshold_is_inclusive():
    probabilities = _probabilities(
        [0.5]
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
    )

    assert len(episodes) == 1


def test_early_detection():
    probabilities = _probabilities(
        [0.8],
        start="2020-01-01 06:00",
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert episodes.loc[
        0, "classification"
    ] == "early_detection"
    assert episodes.loc[
        0, "associated_event_id"
    ] == 1
    assert episodes.loc[
        0, "lead_time"
    ] == pd.Timedelta(
        hours=4
    )


def test_early_detection_left_boundary_inclusive():
    probabilities = _probabilities(
        [0.8],
        start="2020-01-01 04:00",
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert episodes.loc[
        0, "classification"
    ] == "early_detection"
    assert episodes.loc[
        0, "lead_time"
    ] == pd.Timedelta(
        hours=6
    )


def test_storm_start_is_late_not_early():
    probabilities = _probabilities(
        [0.8],
        start="2020-01-01 10:00",
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert episodes.loc[
        0, "classification"
    ] == "late_detection"
    assert pd.isna(
        episodes.loc[
            0, "lead_time"
        ]
    )


def test_late_detection_end_boundary_inclusive():
    probabilities = _probabilities(
        [0.8],
        start="2020-01-01 15:00",
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert episodes.loc[
        0, "classification"
    ] == "late_detection"


def test_false_alarm():
    probabilities = _probabilities(
        [0.8],
        start="2020-01-01 01:00",
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert episodes.loc[
        0, "classification"
    ] == "false_alarm"
    assert pd.isna(
        episodes.loc[
            0, "associated_event_id"
        ]
    )


def test_episode_cannot_detect_multiple_storms_ambiguously():
    probabilities = _probabilities(
        [0.8],
        start="2020-01-01 08:00",
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 11:00"
                ),
                "complete",
            ],
            [
                2,
                pd.Timestamp(
                    "2020-01-01 12:00"
                ),
                pd.Timestamp(
                    "2020-01-01 14:00"
                ),
                "complete",
            ],
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    # Chronological deterministic association.
    assert episodes.loc[
        0, "associated_event_id"
    ] == 1


def test_multiple_alerts_single_storm_do_not_double_count_recall():
    probabilities = pd.Series(
        [0.8, 0.8],
        index=pd.DatetimeIndex(
            [
                "2020-01-01 04:00",
                "2020-01-01 08:00",
            ]
        ),
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert len(episodes) == 2
    assert event_recall(
        episodes,
        events,
    ) == 1.0


def test_multiple_early_episodes_uses_earliest_for_lead_time():
    probabilities = pd.Series(
        [0.8, 0.8],
        index=pd.DatetimeIndex(
            [
                "2020-01-01 04:00",
                "2020-01-01 08:00",
            ]
        ),
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    lead_times = (
        early_detection_lead_times(
            episodes
        )
    )

    assert len(lead_times) == 1
    assert lead_times.iloc[
        0
    ] == pd.Timedelta(
        hours=6
    )


def test_early_episode_preferred_for_storm_level_lead_time():
    probabilities = pd.Series(
        [0.8, 0.8],
        index=pd.DatetimeIndex(
            [
                "2020-01-01 08:00",
                "2020-01-01 12:00",
            ]
        ),
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert set(
        episodes[
            "classification"
        ]
    ) == {
        "early_detection",
        "late_detection",
    }

    lead_times = (
        early_detection_lead_times(
            episodes
        )
    )

    assert lead_times.tolist() == [
        pd.Timedelta(
            hours=2
        )
    ]


def test_multiple_storms():
    probabilities = pd.Series(
        [0.8, 0.8],
        index=pd.DatetimeIndex(
            [
                "2020-01-01 08:00",
                "2020-01-02 08:00",
            ]
        ),
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ],
            [
                2,
                pd.Timestamp(
                    "2020-01-02 10:00"
                ),
                pd.Timestamp(
                    "2020-01-02 15:00"
                ),
                "complete",
            ],
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert set(
        episodes[
            "associated_event_id"
        ].dropna()
    ) == {
        1,
        2,
    }

    assert event_recall(
        episodes,
        events,
    ) == 1.0


def test_event_recall_partial_detection():
    probabilities = _probabilities(
        [0.8],
        start="2020-01-01 08:00",
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.Timestamp(
                    "2020-01-01 15:00"
                ),
                "complete",
            ],
            [
                2,
                pd.Timestamp(
                    "2020-01-02 10:00"
                ),
                pd.Timestamp(
                    "2020-01-02 15:00"
                ),
                "complete",
            ],
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert event_recall(
        episodes,
        events,
    ) == 0.5


def test_valid_prediction_exposure():
    probabilities = _probabilities(
        [0.1] * 12
        + [np.nan] * 12
    )

    assert (
        valid_prediction_exposure_days(
            probabilities
        )
        == 0.5
    )


def test_false_alarm_counted_per_episode():
    probabilities = _probabilities(
        [0.8, 0.9]
        + [0.1] * 22
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
    )

    events = _events([])

    episodes = (
        associate_alerts_with_events(
            episodes,
            events,
        )
    )

    assert len(episodes) == 1

    assert false_alarm_rate_per_day(
        episodes,
        probabilities,
    ) == 1.0


def test_far_uses_valid_prediction_exposure():
    probabilities = _probabilities(
        [0.8]
        + [0.1] * 11
        + [np.nan] * 12
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
    )

    episodes = (
        associate_alerts_with_events(
            episodes,
            _events([]),
        )
    )

    # 1 false episode / 0.5 valid prediction days.
    assert false_alarm_rate_per_day(
        episodes,
        probabilities,
    ) == 2.0


def test_zero_valid_exposure_returns_nan():
    probabilities = _probabilities(
        [np.nan] * 24
    )

    episodes = identify_alerts(
        probabilities,
        0.5,
    )

    assert np.isnan(
        false_alarm_rate_per_day(
            episodes,
            probabilities,
        )
    )


def test_right_censored_event_allows_early_detection():
    probabilities = _probabilities(
        [0.8],
        start="2020-01-01 08:00",
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.NaT,
                "right_censored",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert episodes.loc[
        0, "classification"
    ] == "early_detection"


def test_right_censored_event_does_not_invent_late_window():
    probabilities = _probabilities(
        [0.8],
        start="2020-01-01 12:00",
    )

    events = _events(
        [
            [
                1,
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.NaT,
                "right_censored",
            ]
        ]
    )

    episodes = _associate(
        probabilities,
        events,
    )

    assert episodes.loc[
        0, "classification"
    ] == "false_alarm"


def test_invalid_probability_raises():
    probabilities = _probabilities(
        [0.1, 1.2]
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        identify_alerts(
            probabilities,
            0.5,
        )


def test_malformed_probability_raises():
    probabilities = pd.Series(
        [0.1, "invalid"],
        index=pd.date_range(
            "2020-01-01",
            periods=2,
            freq="h",
        ),
    )

    with pytest.raises(
        ValueError,
        match="numeric",
    ):
        identify_alerts(
            probabilities,
            0.5,
        )


def test_duplicate_prediction_timestamp_raises():
    probabilities = pd.Series(
        [0.1, 0.8],
        index=pd.DatetimeIndex(
            [
                "2020-01-01 00:00",
                "2020-01-01 00:00",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        identify_alerts(
            probabilities,
            0.5,
        )


def test_nonmonotonic_predictions_raise():
    probabilities = pd.Series(
        [0.1, 0.8],
        index=pd.DatetimeIndex(
            [
                "2020-01-01 01:00",
                "2020-01-01 00:00",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="monotonically",
    ):
        identify_alerts(
            probabilities,
            0.5,
        )


def test_non_hour_aligned_prediction_raises():
    probabilities = pd.Series(
        [0.8],
        index=pd.DatetimeIndex(
            [
                "2020-01-01 00:30",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="whole hours",
    ):
        identify_alerts(
            probabilities,
            0.5,
        )


def test_invalid_threshold_raises():
    probabilities = _probabilities(
        [0.8]
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        identify_alerts(
            probabilities,
            1.5,
        )


def test_empty_events_produces_false_alarm():
    probabilities = _probabilities(
        [0.8]
    )

    episodes = _associate(
        probabilities,
        _events([]),
    )

    assert episodes.loc[
        0, "classification"
    ] == "false_alarm"


def test_no_events_recall_is_nan():
    episodes = identify_alerts(
        _probabilities(
            [0.8]
        ),
        0.5,
    )

    assert np.isnan(
        event_recall(
            episodes,
            _events([]),
        )
    )