import numpy as np
import pandas as pd
import pytest

from src.definitions.events import (
    identify_events,
)


def _intervals(
    values,
    *,
    start="2020-01-01 00:00",
):
    """Build contiguous canonical 3-hour Kp intervals."""

    starts = pd.date_range(
        start,
        periods=len(values),
        freq="3h",
    )

    return pd.DataFrame(
        {
            "interval_start": starts,
            "interval_end": (
                starts
                + pd.Timedelta(
                    hours=3
                )
            ),
            "kp": values,
        }
    )


def test_basic_event_start():
    kp = _intervals(
        [
            2.0,
            5.0,
            6.0,
            2.0,
            2.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 1

    assert events.loc[
        0,
        "start_time",
    ] == pd.Timestamp(
        "2020-01-01 03:00"
    )


def test_event_start_matches_canonical_kp_interval_start():
    kp = _intervals(
        [
            1.0,
            4.7,
            5.3,
            2.0,
            2.0,
        ]
    )

    events = identify_events(kp)

    assert events.loc[
        0,
        "start_time",
    ] == kp.loc[
        2,
        "interval_start",
    ]


def test_repeated_hourly_kp_rows_do_not_create_multiple_events():
    kp = _intervals(
        [
            1.0,
            5.0,
            1.0,
            1.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 1


def test_event_termination_after_z_hours():
    kp = _intervals(
        [
            1.0,
            5.0,
            2.0,
            2.0,
            5.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 2

    first = events.iloc[0]

    assert first["start_time"] == pd.Timestamp(
        "2020-01-01 03:00"
    )

    # Storm interval is [03:00, 06:00).
    # Quiet run begins at 06:00.
    # Event's final active hourly state is 05:00.
    assert first["end_time"] == pd.Timestamp(
        "2020-01-01 05:00"
    )

    assert (
        first["boundary_status"]
        == "complete"
    )


def test_same_event_when_gap_less_than_z():
    # One quiet 3-hour interval is only 3 quiet hours.
    kp = _intervals(
        [
            1.0,
            5.0,
            2.0,
            6.0,
            2.0,
            2.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 1

    assert events.loc[
        0,
        "start_time",
    ] == pd.Timestamp(
        "2020-01-01 03:00"
    )

    assert events.loc[
        0,
        "peak_kp",
    ] == 6.0


def test_new_event_after_z_hour_separation():
    kp = _intervals(
        [
            1.0,
            5.0,
            2.0,
            2.0,
            6.0,
            2.0,
            2.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 2

    assert events.loc[
        0,
        "start_time",
    ] == pd.Timestamp(
        "2020-01-01 03:00"
    )

    assert events.loc[
        1,
        "start_time",
    ] == pd.Timestamp(
        "2020-01-01 12:00"
    )


def test_consecutive_storm_intervals_are_one_event():
    kp = _intervals(
        [
            1.0,
            5.0,
            6.0,
            7.0,
            2.0,
            2.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 1

    assert events.loc[
        0,
        "peak_kp",
    ] == 7.0


def test_missing_kp_does_not_count_as_below_threshold():
    kp = _intervals(
        [
            1.0,
            5.0,
            2.0,
            np.nan,
            2.0,
            5.0,
            2.0,
            2.0,
        ]
    )

    events = identify_events(kp)

    # The two quiet intervals separated by missing Kp
    # must not combine into a 6-hour quiet run.
    assert len(events) == 1

    assert events.loc[
        0,
        "start_time",
    ] == pd.Timestamp(
        "2020-01-01 03:00"
    )

    assert events.loc[
        0,
        "peak_kp",
    ] == 5.0


def test_missing_canonical_interval_does_not_count_as_quiet():
    kp = _intervals(
        [
            1.0,
            5.0,
            2.0,
            2.0,
            5.0,
            2.0,
            2.0,
        ]
    )

    # Remove the interval beginning at 09:00.
    # This creates a real 3-hour source gap.
    kp = kp.drop(
        index=3
    ).reset_index(
        drop=True
    )

    events = identify_events(kp)

    # 06:00-09:00 quiet + missing 09:00-12:00 +
    # storm at 12:00 must remain the same event.
    assert len(events) == 1

    assert events.loc[
        0,
        "start_time",
    ] == pd.Timestamp(
        "2020-01-01 03:00"
    )


def test_missing_state_resets_partial_quiet_run():
    kp = _intervals(
        [
            5.0,
            2.0,
            np.nan,
            2.0,
            5.0,
            2.0,
            2.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 1

    assert events.loc[
        0,
        "peak_kp",
    ] == 5.0


def test_left_dataset_boundary_is_censored():
    kp = _intervals(
        [
            6.0,
            2.0,
            2.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 1

    event = events.iloc[0]

    assert event[
        "start_time"
    ] == pd.Timestamp(
        "2020-01-01 00:00"
    )

    assert (
        event["boundary_status"]
        == "left_censored"
    )

    assert event[
        "end_time"
    ] == pd.Timestamp(
        "2020-01-01 02:00"
    )


def test_right_dataset_boundary_is_censored():
    kp = _intervals(
        [
            1.0,
            5.0,
            6.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 1

    event = events.iloc[0]

    assert pd.isna(
        event["end_time"]
    )

    assert (
        event["boundary_status"]
        == "right_censored"
    )


def test_both_dataset_boundaries_are_censored():
    kp = _intervals(
        [
            6.0,
            7.0,
        ]
    )

    events = identify_events(kp)

    assert len(events) == 1

    event = events.iloc[0]

    assert pd.isna(
        event["end_time"]
    )

    assert (
        event["boundary_status"]
        == "both_censored"
    )


def test_quiet_dataset_has_no_events():
    kp = _intervals(
        [
            1.0,
            2.0,
            4.7,
            3.0,
        ]
    )

    events = identify_events(kp)

    assert events.empty

    assert list(
        events.columns
    ) == [
        "event_id",
        "start_time",
        "end_time",
        "threshold",
        "peak_kp",
        "boundary_status",
    ]


def test_all_missing_dataset_has_no_events():
    kp = _intervals(
        [
            np.nan,
            np.nan,
        ]
    )

    events = identify_events(kp)

    assert events.empty


def test_custom_threshold():
    kp = _intervals(
        [
            1.0,
            6.0,
            2.0,
            2.0,
        ]
    )

    events = identify_events(
        kp,
        threshold=7.0,
    )

    assert events.empty


def test_peak_kp_excludes_quiet_termination_run():
    kp = _intervals(
        [
            5.0,
            6.0,
            4.0,
            4.0,
        ]
    )

    events = identify_events(kp)

    assert events.loc[
        0,
        "peak_kp",
    ] == 6.0


def test_invalid_interval_duration_raises():
    kp = _intervals(
        [
            1.0,
            5.0,
        ]
    )

    kp.loc[
        1,
        "interval_end",
    ] = (
        kp.loc[
            1,
            "interval_start",
        ]
        + pd.Timedelta(
            hours=2
        )
    )

    with pytest.raises(
        ValueError,
        match="exactly 3 hours",
    ):
        identify_events(kp)


def test_overlapping_intervals_raise():
    kp = _intervals(
        [
            1.0,
            5.0,
        ]
    )

    kp.loc[
        1,
        "interval_start",
    ] = pd.Timestamp(
        "2020-01-01 02:00"
    )

    kp.loc[
        1,
        "interval_end",
    ] = pd.Timestamp(
        "2020-01-01 05:00"
    )

    with pytest.raises(
        ValueError,
        match="must not overlap",
    ):
        identify_events(kp)


def test_duplicate_interval_start_raises():
    kp = _intervals(
        [
            1.0,
            5.0,
        ]
    )

    kp.loc[
        1,
        "interval_start",
    ] = kp.loc[
        0,
        "interval_start",
    ]

    kp.loc[
        1,
        "interval_end",
    ] = kp.loc[
        0,
        "interval_end",
    ]

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        identify_events(kp)


def test_nonmonotonic_intervals_raise():
    kp = _intervals(
        [
            1.0,
            5.0,
            2.0,
        ]
    )

    kp = kp.iloc[
        [0, 2, 1]
    ].reset_index(
        drop=True
    )

    with pytest.raises(
        ValueError,
        match="ordered",
    ):
        identify_events(kp)


def test_missing_required_column_raises():
    kp = _intervals(
        [
            1.0,
            5.0,
        ]
    ).drop(
        columns="interval_end"
    )

    with pytest.raises(
        ValueError,
        match="required columns",
    ):
        identify_events(kp)


def test_invalid_termination_hours_raises():
    kp = _intervals(
        [
            1.0,
            5.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        identify_events(
            kp,
            termination_hours=0,
        )