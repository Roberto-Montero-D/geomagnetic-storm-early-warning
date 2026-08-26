import numpy as np
import pandas as pd
import pytest

import src.final_test.prediction as phase8_prediction
from src.final_test.contract import (
    PHASE8_FEATURES,
    PHASE8_MODEL_CONFIG_ID,
    PHASE8_OPERATIONAL_THRESHOLD,
)
from src.final_test.materialization import (
    Phase8Materialization,
    materialize_phase8_data,
)


def _frames():
    index = pd.DatetimeIndex(
        [
            "2021-12-31 21:00",
            "2021-12-31 22:00",
            "2021-12-31 23:00",
            "2022-01-01 00:00",
            "2022-01-01 01:00",
            "2022-01-01 02:00",
        ],
        name="prediction_time",
    )

    values = {
        feature: np.arange(len(index), dtype=float) + i
        for i, feature in enumerate(PHASE8_FEATURES)
    }
    values["target"] = [0.0, 1.0, 0.0, 1.0, np.nan, 0.0]
    dataset = pd.DataFrame(values, index=index)

    status = pd.DataFrame(
        {
            "supervised_eligible": [
                True,
                True,
                True,
                True,
                False,
                True,
            ],
            "features_complete": [
                True,
                True,
                True,
                True,
                True,
                False,
            ],
        },
        index=index,
    )

    splits = pd.DataFrame(
        {
            "period": [
                "validation_3",
                "validation_3",
                "validation_3",
                "final_test",
                "final_test",
                "final_test",
            ]
        },
        index=index,
    )

    # Make the last Final Test row feature-incomplete.
    dataset.loc[index[-1], PHASE8_FEATURES[0]] = np.nan

    return dataset, status, splits


def test_materialization_trains_pre2022_only():
    dataset, status, splits = _frames()

    result = materialize_phase8_data(
        dataset,
        status,
        splits,
    )

    assert (result.train_index < pd.Timestamp("2022-01-01")).all()
    assert len(result.x_train) == 3
    assert result.y_train.tolist() == [0.0, 1.0, 0.0]


def test_final_test_prediction_rows_do_not_depend_on_target_known():
    dataset, status, splits = _frames()

    result = materialize_phase8_data(
        dataset,
        status,
        splits,
    )

    # 00:00 has known target, 01:00 has unknown target. Both are predicted
    # because both have complete features. 02:00 is excluded only because its
    # features are incomplete.
    assert result.final_test_index.tolist() == [
        pd.Timestamp("2022-01-01 00:00"),
        pd.Timestamp("2022-01-01 01:00"),
    ]

    mutated = dataset.copy()
    mutated.loc[
        pd.Timestamp("2022-01-01 00:00"),
        "target",
    ] = 0.0
    mutated.loc[
        pd.Timestamp("2022-01-01 01:00"),
        "target",
    ] = 1.0

    again = materialize_phase8_data(
        mutated,
        status,
        splits,
    )

    pd.testing.assert_frame_equal(
        result.x_final_test,
        again.x_final_test,
    )


def test_materialization_exposes_only_frozen_features():
    dataset, status, splits = _frames()
    dataset["future_metadata"] = 123

    result = materialize_phase8_data(
        dataset,
        status,
        splits,
    )

    assert tuple(result.x_train.columns) == PHASE8_FEATURES
    assert tuple(result.x_final_test.columns) == PHASE8_FEATURES
    assert "target" not in result.x_final_test.columns
    assert "future_metadata" not in result.x_final_test.columns


def test_final_test_targets_are_not_returned():
    dataset, status, splits = _frames()

    result = materialize_phase8_data(
        dataset,
        status,
        splits,
    )

    assert not hasattr(result, "y_final_test")
    assert not hasattr(result, "target")


def test_rejects_split_calendar_disagreement():
    dataset, status, splits = _frames()
    splits.loc[
        pd.Timestamp("2022-01-01 00:00"),
        "period",
    ] = "validation_3"

    with pytest.raises(
        AssertionError,
        match="calendar disagrees",
    ):
        materialize_phase8_data(
            dataset,
            status,
            splits,
        )


class _FakeFrozenModel:
    def __init__(self):
        self.fit_x = None
        self.fit_y = None
        self.predict_x = None

    def fit(self, x, y):
        self.fit_x = x.copy()
        self.fit_y = y.copy()
        return self

    def predict_proba(self, x):
        self.predict_x = x.copy()
        p = np.linspace(0.2, 0.8, len(x))
        return np.column_stack([1.0 - p, p])


def test_prediction_uses_exactly_one_frozen_model(monkeypatch):
    dataset, status, splits = _frames()
    materialized = materialize_phase8_data(
        dataset,
        status,
        splits,
    )

    calls = []
    fake = _FakeFrozenModel()

    def factory(config_id):
        calls.append(config_id)
        return fake

    monkeypatch.setattr(
        phase8_prediction,
        "make_phase5_model_by_id",
        factory,
    )

    result = phase8_prediction.generate_phase8_predictions(
        materialized,
    )

    assert calls == [PHASE8_MODEL_CONFIG_ID]
    assert result.config_id == PHASE8_MODEL_CONFIG_ID
    assert result.operational_threshold == PHASE8_OPERATIONAL_THRESHOLD
    pd.testing.assert_frame_equal(
        fake.fit_x,
        materialized.x_train,
    )
    pd.testing.assert_series_equal(
        fake.fit_y,
        materialized.y_train,
    )
    pd.testing.assert_frame_equal(
        fake.predict_x,
        materialized.x_final_test,
    )


def test_prediction_artifact_contains_probability_only(monkeypatch):
    dataset, status, splits = _frames()
    materialized = materialize_phase8_data(
        dataset,
        status,
        splits,
    )

    monkeypatch.setattr(
        phase8_prediction,
        "make_phase5_model_by_id",
        lambda config_id: _FakeFrozenModel(),
    )

    result = phase8_prediction.generate_phase8_predictions(
        materialized,
    )

    assert tuple(result.table.columns) == ("probability",)
    assert "target" not in result.table.columns
    assert "alert" not in result.table.columns
    assert "storm_id" not in result.table.columns
    assert result.table.index.equals(
        materialized.x_final_test.index
    )


def test_prediction_module_has_no_threshold_search_api():
    assert not hasattr(
        phase8_prediction,
        "optimize_phase6_threshold",
    )
    assert not hasattr(
        phase8_prediction,
        "DEFAULT_THRESHOLD_GRID",
    )
