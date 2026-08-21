import pandas as pd
import pytest

from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    PERIOD_INITIAL_TRAIN,
    PERIOD_OUTSIDE_PRIMARY,
    PERIOD_VALIDATION_1,
    PERIOD_VALIDATION_2,
    PERIOD_VALIDATION_3,
    assign_temporal_periods,
    development_fold_masks,
)


def test_exact_atomic_boundaries():
    times = pd.DatetimeIndex(
        [
            "1995-12-31 23:00",
            "1996-01-01 00:00",
            "2016-12-31 23:00",
            "2017-01-01 00:00",
            "2018-12-31 23:00",
            "2019-01-01 00:00",
            "2020-12-31 23:00",
            "2021-01-01 00:00",
            "2021-12-31 23:00",
            "2022-01-01 00:00",
            "2025-12-31 23:00",
            "2026-01-01 00:00",
        ],
        name="prediction_time",
    )

    split = assign_temporal_periods(times)

    assert split["period"].tolist() == [
        PERIOD_OUTSIDE_PRIMARY,
        PERIOD_INITIAL_TRAIN,
        PERIOD_INITIAL_TRAIN,
        PERIOD_VALIDATION_1,
        PERIOD_VALIDATION_1,
        PERIOD_VALIDATION_2,
        PERIOD_VALIDATION_2,
        PERIOD_VALIDATION_3,
        PERIOD_VALIDATION_3,
        PERIOD_FINAL_TEST,
        PERIOD_FINAL_TEST,
        PERIOD_OUTSIDE_PRIMARY,
    ]


def test_final_test_flag_only_for_2022_through_2025():
    times = pd.DatetimeIndex(
        [
            "2021-12-31 23:00",
            "2022-01-01 00:00",
            "2025-12-31 23:00",
            "2026-01-01 00:00",
        ]
    )

    split = assign_temporal_periods(times)

    assert split["is_final_test"].tolist() == [
        False, True, True, False
    ]


def test_primary_coverage_flag_matches_frozen_calendar():
    times = pd.DatetimeIndex(
        [
            "1995-12-31 23:00",
            "1996-01-01 00:00",
            "2025-12-31 23:00",
            "2026-01-01 00:00",
        ]
    )

    split = assign_temporal_periods(times)
    assert split["in_primary_coverage"].tolist() == [
        False, True, True, False
    ]


def test_screening_fold_is_1996_2016_to_2017_2018():
    times = pd.DatetimeIndex(
        [
            "2016-12-31 23:00",
            "2017-01-01 00:00",
            "2018-12-31 23:00",
            "2019-01-01 00:00",
        ]
    )
    split = assign_temporal_periods(times)
    fold = development_fold_masks(split)["screening"]

    assert fold["train"].tolist() == [True, False, False, False]
    assert fold["validation"].tolist() == [False, True, True, False]


def test_walk_forward_1_expands_training_through_2018():
    times = pd.DatetimeIndex(
        [
            "1996-01-01 00:00",
            "2017-01-01 00:00",
            "2019-01-01 00:00",
            "2021-01-01 00:00",
        ]
    )
    split = assign_temporal_periods(times)
    fold = development_fold_masks(split)["walk_forward_1"]

    assert fold["train"].tolist() == [True, True, False, False]
    assert fold["validation"].tolist() == [False, False, True, False]


def test_walk_forward_2_expands_training_through_2020():
    times = pd.DatetimeIndex(
        [
            "1996-01-01 00:00",
            "2017-01-01 00:00",
            "2019-01-01 00:00",
            "2021-01-01 00:00",
            "2022-01-01 00:00",
        ]
    )
    split = assign_temporal_periods(times)
    fold = development_fold_masks(split)["walk_forward_2"]

    assert fold["train"].tolist() == [True, True, True, False, False]
    assert fold["validation"].tolist() == [False, False, False, True, False]


def test_final_test_is_in_no_development_mask():
    times = pd.date_range(
        "2021-12-31 22:00",
        "2022-01-01 03:00",
        freq="h",
    )
    split = assign_temporal_periods(times)
    folds = development_fold_masks(split)
    final = split["is_final_test"]

    for fold in folds.values():
        assert not (fold["train"] & final).any()
        assert not (fold["validation"] & final).any()


def test_split_assignment_preserves_index():
    times = pd.date_range(
        "2018-12-31 22:00",
        periods=6,
        freq="h",
        name="prediction_time",
    )
    split = assign_temporal_periods(times)
    pd.testing.assert_index_equal(split.index, times)


@pytest.mark.parametrize(
    "times",
    [
        ["2020-01-01 00:30"],
        ["2020-01-01 01:00", "2020-01-01 00:00"],
        ["2020-01-01 00:00", "2020-01-01 00:00"],
    ],
)
def test_invalid_prediction_times_raise(times):
    with pytest.raises(ValueError):
        assign_temporal_periods(pd.DatetimeIndex(times))


def test_timezone_aware_times_raise():
    times = pd.date_range(
        "2020-01-01",
        periods=2,
        freq="h",
        tz="UTC",
    )
    with pytest.raises(ValueError, match="timezone-naive"):
        assign_temporal_periods(times)


def test_fold_builder_requires_period_column():
    with pytest.raises(ValueError, match="period"):
        development_fold_masks(pd.DataFrame({"x": [1]}))
