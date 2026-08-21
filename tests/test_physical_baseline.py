import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import build_development_folds
from src.baselines.physical import (
    PHYSICAL_FEATURES,
    predict_physical,
    predict_physical_for_index,
)
from src.dataset.row_status import build_row_status
from src.dataset.temporal_splits import assign_temporal_periods
from src.features.integrated import PRIMARY_FEATURE_COLUMNS


def test_frozen_physical_feature_names():
    assert PHYSICAL_FEATURES == ("bz_gsm", "speed")


def test_rule_uses_strict_bz_and_speed_inequalities():
    frame = pd.DataFrame(
        {
            "bz_gsm": [-5.01, -5.00, -5.01, -6.0],
            "speed": [500.01, 500.01, 500.00, 600.0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="h"),
    )

    prediction = predict_physical(
        frame,
        bz_magnitude_nt=5.0,
        speed_threshold_km_s=500.0,
    )

    assert prediction.tolist() == [1, 0, 0, 1]


def test_either_condition_false_produces_negative():
    frame = pd.DataFrame(
        {
            "bz_gsm": [-6.0, -4.0, -4.0],
            "speed": [400.0, 600.0, 400.0],
        }
    )

    prediction = predict_physical(
        frame,
        bz_magnitude_nt=5.0,
        speed_threshold_km_s=500.0,
    )

    assert prediction.tolist() == [0, 0, 0]


def test_missing_required_feature_produces_missing_prediction():
    frame = pd.DataFrame(
        {
            "bz_gsm": [-6.0, np.nan, -6.0],
            "speed": [600.0, 600.0, np.nan],
        }
    )

    prediction = predict_physical(
        frame,
        bz_magnitude_nt=5.0,
        speed_threshold_km_s=500.0,
    )

    assert prediction.iloc[0] == 1
    assert pd.isna(prediction.iloc[1])
    assert pd.isna(prediction.iloc[2])


def test_target_and_future_like_columns_cannot_change_prediction():
    index = pd.date_range("2020-01-01", periods=2, freq="h")
    base = pd.DataFrame(
        {
            "bz_gsm": [-6.0, -4.0],
            "speed": [600.0, 600.0],
        },
        index=index,
    )
    contaminated = base.assign(
        target=[1.0, 0.0],
        future_kp=[9.0, 0.0],
        storm_id=[99, 100],
    )

    a = predict_physical(
        base,
        bz_magnitude_nt=5.0,
        speed_threshold_km_s=500.0,
    )
    b = predict_physical(
        contaminated,
        bz_magnitude_nt=5.0,
        speed_threshold_km_s=500.0,
    )

    pd.testing.assert_series_equal(a, b)


def test_thresholds_are_mandatory():
    frame = pd.DataFrame(
        {"bz_gsm": [-6.0], "speed": [600.0]}
    )

    with pytest.raises(TypeError):
        predict_physical(frame)


@pytest.mark.parametrize(
    ("bz_threshold", "speed_threshold"),
    [
        (0.0, 500.0),
        (-5.0, 500.0),
        (5.0, 0.0),
        (5.0, -500.0),
        (np.inf, 500.0),
        (5.0, np.nan),
    ],
)
def test_invalid_thresholds_raise(
    bz_threshold,
    speed_threshold,
):
    frame = pd.DataFrame(
        {"bz_gsm": [-6.0], "speed": [600.0]}
    )

    with pytest.raises(ValueError):
        predict_physical(
            frame,
            bz_magnitude_nt=bz_threshold,
            speed_threshold_km_s=speed_threshold,
        )


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="required physical"):
        predict_physical(
            pd.DataFrame({"bz_gsm": [-6.0]}),
            bz_magnitude_nt=5.0,
            speed_threshold_km_s=500.0,
        )


def _canonical_like_dataset():
    times = pd.DatetimeIndex(
        [
            "2016-12-31 23:00",
            "2017-01-01 00:00",
            "2018-12-31 23:00",
            "2019-01-01 00:00",
            "2020-12-31 23:00",
            "2021-01-01 00:00",
            "2021-12-31 23:00",
            "2022-01-01 00:00",
        ],
        name="prediction_time",
    )
    frame = pd.DataFrame(
        1.0,
        index=times,
        columns=list(PRIMARY_FEATURE_COLUMNS),
    )
    frame["bz_gsm"] = [-6, -4, -7, -8, -3, -9, -2, -20]
    frame["speed"] = [600, 600, 400, 700, 300, 800, 900, 1000]
    frame["target"] = [0, 1, 0, 1, 0, 1, 0, 1]
    return frame


def test_b1_applies_to_materialized_development_validation_index():
    dataset = _canonical_like_dataset()
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)
    fold = build_development_folds(
        dataset, status, splits
    )["walk_forward_2"]

    prediction = predict_physical_for_index(
        dataset,
        fold.validation_index,
        bz_magnitude_nt=5.0,
        speed_threshold_km_s=500.0,
    )

    pd.testing.assert_index_equal(
        prediction.index,
        fold.validation_index,
    )
    assert pd.Timestamp("2022-01-01 00:00") not in prediction.index


def test_final_test_physical_values_cannot_affect_development_prediction():
    dataset = _canonical_like_dataset()
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)
    fold = build_development_folds(
        dataset, status, splits
    )["walk_forward_2"]

    before = predict_physical_for_index(
        dataset,
        fold.validation_index,
        bz_magnitude_nt=5.0,
        speed_threshold_km_s=500.0,
    )

    mutated = dataset.copy()
    final_time = pd.Timestamp("2022-01-01 00:00")
    mutated.loc[final_time, "bz_gsm"] = 20.0
    mutated.loc[final_time, "speed"] = 0.0

    after = predict_physical_for_index(
        mutated,
        fold.validation_index,
        bz_magnitude_nt=5.0,
        speed_threshold_km_s=500.0,
    )

    pd.testing.assert_series_equal(before, after)


def test_index_wrapper_rejects_unknown_timestamps():
    dataset = _canonical_like_dataset()

    with pytest.raises(ValueError, match="absent"):
        predict_physical_for_index(
            dataset,
            pd.DatetimeIndex(["2030-01-01 00:00"]),
            bz_magnitude_nt=5.0,
            speed_threshold_km_s=500.0,
        )
