import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import build_development_folds
from src.baselines.persistence import (
    DEFAULT_STORM_THRESHOLD,
    PERSISTENCE_FEATURE,
    predict_persistence,
    predict_persistence_for_index,
)
from src.dataset.row_status import build_row_status
from src.dataset.temporal_splits import assign_temporal_periods
from src.features.integrated import PRIMARY_FEATURE_COLUMNS


def test_frozen_feature_and_threshold():
    assert PERSISTENCE_FEATURE == "kp_lag_1h"
    assert DEFAULT_STORM_THRESHOLD == 5.0


def test_threshold_rule_is_inclusive():
    frame = pd.DataFrame(
        {PERSISTENCE_FEATURE: [4.99, 5.0, 5.01]},
        index=pd.date_range("2020-01-01", periods=3, freq="h"),
    )

    prediction = predict_persistence(frame)

    assert prediction.tolist() == [0, 1, 1]


def test_missing_kp_remains_missing_not_negative():
    frame = pd.DataFrame(
        {PERSISTENCE_FEATURE: [4.0, np.nan, 6.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="h"),
    )

    prediction = predict_persistence(frame)

    assert prediction.iloc[0] == 0
    assert pd.isna(prediction.iloc[1])
    assert prediction.iloc[2] == 1


def test_prediction_does_not_read_target():
    index = pd.date_range("2020-01-01", periods=2, freq="h")
    a = pd.DataFrame(
        {
            PERSISTENCE_FEATURE: [4.0, 6.0],
            "target": [0.0, 0.0],
        },
        index=index,
    )
    b = a.copy()
    b["target"] = [1.0, np.nan]

    pd.testing.assert_series_equal(
        predict_persistence(a),
        predict_persistence(b),
    )


def test_unrelated_or_future_like_columns_cannot_change_prediction():
    index = pd.date_range("2020-01-01", periods=2, freq="h")
    a = pd.DataFrame(
        {PERSISTENCE_FEATURE: [4.0, 6.0]},
        index=index,
    )
    b = a.assign(
        target=[1.0, 0.0],
        future_kp=[9.0, 0.0],
        future_window_positive=[True, False],
    )

    pd.testing.assert_series_equal(
        predict_persistence(a),
        predict_persistence(b),
    )


def test_custom_threshold_is_explicit_and_deterministic():
    frame = pd.DataFrame(
        {PERSISTENCE_FEATURE: [3.9, 4.0, 4.1]},
        index=pd.date_range("2020-01-01", periods=3, freq="h"),
    )

    assert predict_persistence(
        frame, threshold=4.0
    ).tolist() == [0, 1, 1]


def test_missing_persistence_feature_raises():
    with pytest.raises(ValueError, match="kp_lag_1h"):
        predict_persistence(pd.DataFrame({"x": [1.0]}))


def test_nonpositive_threshold_raises():
    frame = pd.DataFrame({PERSISTENCE_FEATURE: [5.0]})

    with pytest.raises(ValueError, match="positive"):
        predict_persistence(frame, threshold=0.0)


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
    frame[PERSISTENCE_FEATURE] = [
        4.0, 5.0, 6.0, 4.0, 7.0, 5.0, 3.0, 9.0
    ]
    frame["target"] = [0, 1, 0, 1, 0, 1, 0, 1]
    return frame


def test_b0_applies_to_materialized_development_validation_index():
    dataset = _canonical_like_dataset()
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)
    fold = build_development_folds(
        dataset, status, splits
    )["walk_forward_2"]

    prediction = predict_persistence_for_index(
        dataset,
        fold.validation_index,
    )

    pd.testing.assert_index_equal(
        prediction.index,
        fold.validation_index,
    )
    assert pd.Timestamp("2022-01-01 00:00") not in prediction.index


def test_final_test_value_cannot_affect_development_predictions():
    dataset = _canonical_like_dataset()
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)
    fold = build_development_folds(
        dataset, status, splits
    )["walk_forward_2"]

    before = predict_persistence_for_index(
        dataset, fold.validation_index
    )

    mutated = dataset.copy()
    mutated.loc[
        pd.Timestamp("2022-01-01 00:00"),
        PERSISTENCE_FEATURE,
    ] = 0.0

    after = predict_persistence_for_index(
        mutated, fold.validation_index
    )

    pd.testing.assert_series_equal(before, after)


def test_index_wrapper_rejects_unknown_timestamps():
    dataset = _canonical_like_dataset()
    unknown = pd.DatetimeIndex(
        ["2030-01-01 00:00"],
        name="prediction_time",
    )

    with pytest.raises(ValueError, match="absent"):
        predict_persistence_for_index(dataset, unknown)
