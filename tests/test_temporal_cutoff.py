import pandas as pd
import pytest

from src.temporal.cutoff import (
    eligible_interval_mask,
    information_cutoff,
    interval_end_times,
    maximum_eligible_information_time,
    select_eligible_intervals,
)


def _hourly_frame() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01 10:00",
        periods=6,
        freq="h",
    )

    return pd.DataFrame(
        {
            "value": [
                10,
                11,
                12,
                13,
                14,
                15,
            ]
        },
        index=index,
    )


def test_information_cutoff_primary_rule():
    result = information_cutoff(
        pd.Timestamp(
            "2020-01-01 14:00"
        )
    )

    assert result == pd.Timestamp(
        "2020-01-01 13:00"
    )


def test_interval_end_times():
    index = pd.DatetimeIndex(
        [
            "2020-01-01 12:00",
            "2020-01-01 13:00",
        ]
    )

    result = interval_end_times(
        index
    )

    expected = pd.DatetimeIndex(
        [
            "2020-01-01 13:00",
            "2020-01-01 14:00",
        ],
        name="period_end",
    )

    pd.testing.assert_index_equal(
        result,
        expected,
    )


def test_exact_cutoff_boundary_is_eligible():
    index = pd.DatetimeIndex(
        [
            "2020-01-01 12:00",
        ]
    )

    mask = eligible_interval_mask(
        index,
        pd.Timestamp(
            "2020-01-01 14:00"
        ),
    )

    # 12:00 row represents [12:00, 13:00).
    # Prediction cutoff is 13:00.
    assert bool(mask.iloc[0])


def test_interval_ending_after_cutoff_is_ineligible():
    index = pd.DatetimeIndex(
        [
            "2020-01-01 13:00",
        ]
    )

    mask = eligible_interval_mask(
        index,
        pd.Timestamp(
            "2020-01-01 14:00"
        ),
    )

    # 13:00 row represents [13:00, 14:00).
    # It is not available at cutoff 13:00.
    assert not bool(mask.iloc[0])


def test_select_eligible_intervals_primary_example():
    frame = _hourly_frame()

    result = select_eligible_intervals(
        frame,
        pd.Timestamp(
            "2020-01-01 14:00"
        ),
    )

    expected_index = pd.date_range(
        "2020-01-01 10:00",
        "2020-01-01 12:00",
        freq="h",
    )

    pd.testing.assert_index_equal(
        result.index,
        expected_index,
    )


def test_prediction_at_first_hour_can_have_no_history():
    frame = _hourly_frame()

    result = select_eligible_intervals(
        frame,
        pd.Timestamp(
            "2020-01-01 10:00"
        ),
    )

    assert result.empty


def test_maximum_information_time_matches_cutoff():
    frame = _hourly_frame()

    result = (
        maximum_eligible_information_time(
            frame,
            pd.Timestamp(
                "2020-01-01 14:00"
            ),
        )
    )

    assert result == pd.Timestamp(
        "2020-01-01 13:00"
    )


def test_future_values_cannot_enter_selected_history():
    original = _hourly_frame()

    mutated = original.copy()

    # At prediction 14:00, cutoff is 13:00.
    # Rows starting at 13:00 or later are unavailable.
    mutated.loc[
        mutated.index
        >= pd.Timestamp(
            "2020-01-01 13:00"
        ),
        "value",
    ] = 999999

    original_history = (
        select_eligible_intervals(
            original,
            pd.Timestamp(
                "2020-01-01 14:00"
            ),
        )
    )

    mutated_history = (
        select_eligible_intervals(
            mutated,
            pd.Timestamp(
                "2020-01-01 14:00"
            ),
        )
    )

    pd.testing.assert_frame_equal(
        original_history,
        mutated_history,
    )


def test_future_rows_can_be_appended_without_affecting_history():
    frame = _hourly_frame()

    future = pd.DataFrame(
        {
            "value": [
                999,
                1000,
            ]
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-01 16:00",
                "2020-01-01 17:00",
            ]
        ),
    )

    extended = pd.concat(
        [
            frame,
            future,
        ]
    )

    base_history = (
        select_eligible_intervals(
            frame,
            pd.Timestamp(
                "2020-01-01 14:00"
            ),
        )
    )

    extended_history = (
        select_eligible_intervals(
            extended,
            pd.Timestamp(
                "2020-01-01 14:00"
            ),
        )
    )

    pd.testing.assert_frame_equal(
        base_history,
        extended_history,
        check_freq=False,
    )


def test_missing_timestamps_remain_missing():
    frame = pd.DataFrame(
        {
            "value": [
                1,
                3,
            ]
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-01 10:00",
                "2020-01-01 12:00",
            ]
        ),
    )

    result = select_eligible_intervals(
        frame,
        pd.Timestamp(
            "2020-01-01 14:00"
        ),
    )

    assert list(
        result.index
    ) == [
        pd.Timestamp(
            "2020-01-01 10:00"
        ),
        pd.Timestamp(
            "2020-01-01 12:00"
        ),
    ]

    assert (
        pd.Timestamp(
            "2020-01-01 11:00"
        )
        not in result.index
    )


def test_duplicate_observation_times_raise():
    frame = pd.DataFrame(
        {
            "value": [
                1,
                2,
            ]
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-01 10:00",
                "2020-01-01 10:00",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        select_eligible_intervals(
            frame,
            pd.Timestamp(
                "2020-01-01 14:00"
            ),
        )


def test_nonmonotonic_observation_times_raise():
    frame = pd.DataFrame(
        {
            "value": [
                1,
                2,
            ]
        },
        index=pd.DatetimeIndex(
            [
                "2020-01-01 11:00",
                "2020-01-01 10:00",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="monotonically increasing",
    ):
        select_eligible_intervals(
            frame,
            pd.Timestamp(
                "2020-01-01 14:00"
            ),
        )


def test_nat_observation_time_raises():
    frame = pd.DataFrame(
        {
            "value": [
                1,
                2,
            ]
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp(
                    "2020-01-01 10:00"
                ),
                pd.NaT,
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="NaT",
    ):
        select_eligible_intervals(
            frame,
            pd.Timestamp(
                "2020-01-01 14:00"
            ),
        )


def test_nat_prediction_time_raises():
    with pytest.raises(
        ValueError,
        match="prediction_time",
    ):
        information_cutoff(
            pd.NaT
        )


def test_zero_information_delay_raises():
    with pytest.raises(
        ValueError,
        match="information_delay",
    ):
        information_cutoff(
            pd.Timestamp(
                "2020-01-01 14:00"
            ),
            information_delay=pd.Timedelta(
                0
            ),
        )


def test_zero_interval_duration_raises():
    index = pd.DatetimeIndex(
        [
            "2020-01-01 12:00",
        ]
    )

    with pytest.raises(
        ValueError,
        match="interval_duration",
    ):
        interval_end_times(
            index,
            interval_duration=pd.Timedelta(
                0
            ),
        )


def test_custom_interval_duration():
    index = pd.DatetimeIndex(
        [
            "2020-01-01 09:00",
        ]
    )

    mask = eligible_interval_mask(
        index,
        pd.Timestamp(
            "2020-01-01 14:00"
        ),
        interval_duration=pd.Timedelta(
            hours=3
        ),
    )

    # [09:00, 12:00) is completed by cutoff 13:00.
    assert bool(mask.iloc[0])