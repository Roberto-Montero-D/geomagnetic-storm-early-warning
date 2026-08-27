import copy

import numpy as np
import pandas as pd
import pytest

import src.final_test.prediction as phase8_prediction
from src.final_test.contract import (
    PHASE8_FEATURES,
    PHASE8_MODEL_CONFIG_ID,
    PHASE8_OPERATIONAL_THRESHOLD,
)
from src.final_test.materialization import materialize_phase8_data
from src.final_test.prediction import generate_phase8_predictions


def _synthetic_frames():
    index = pd.date_range(
        "2021-12-31 18:00",
        periods=12,
        freq="h",
        name="prediction_time",
    )

    values = {
        feature: np.linspace(
            i,
            i + 1.0,
            len(index),
            dtype=float,
        )
        for i, feature in enumerate(PHASE8_FEATURES)
    }
    values["target"] = [
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        np.nan,
        0.0,
        1.0,
        0.0,
        np.nan,
    ]
    dataset = pd.DataFrame(values, index=index)

    pre2022 = index < pd.Timestamp("2022-01-01")
    final = ~pre2022

    status = pd.DataFrame(
        {
            "supervised_eligible": [
                bool(pre2022[i] and pd.notna(dataset.iloc[i]["target"]))
                for i in range(len(index))
            ],
            "features_complete": [True] * len(index),
        },
        index=index,
    )

    splits = pd.DataFrame(
        {
            "period": [
                "validation_3" if flag else "final_test"
                for flag in pre2022
            ]
        },
        index=index,
    )

    return dataset, status, splits


class _DeterministicFakeModel:
    """Simple deterministic model for synthetic isolation tests."""

    def fit(self, x, y):
        self._offset = float(
            x.to_numpy(dtype=float).mean()
            + y.to_numpy(dtype=float).mean()
        )
        return self

    def predict_proba(self, x):
        raw = x.to_numpy(dtype=float).mean(axis=1)
        score = 1.0 / (
            1.0
            + np.exp(
                -(
                    raw * 0.01
                    + self._offset * 0.001
                )
            )
        )
        return np.column_stack([1.0 - score, score])


def _predict_with_fake(monkeypatch, dataset, status, splits):
    monkeypatch.setattr(
        phase8_prediction,
        "make_phase5_model_by_id",
        lambda config_id: _DeterministicFakeModel(),
    )

    materialized = materialize_phase8_data(
        dataset,
        status,
        splits,
    )
    predictions = generate_phase8_predictions(
        materialized,
    )

    return materialized, predictions


def test_mutating_all_final_test_targets_cannot_change_materialization(
    monkeypatch,
):
    dataset, status, splits = _synthetic_frames()

    base_mat, base_pred = _predict_with_fake(
        monkeypatch,
        dataset,
        status,
        splits,
    )

    mutated = dataset.copy()
    final_mask = splits["period"].eq("final_test")
    final_targets = mutated.loc[final_mask, "target"]

    mutated.loc[final_mask, "target"] = [
        1.0 if pd.isna(value) or value == 0.0 else 0.0
        for value in final_targets
    ]

    new_mat, new_pred = _predict_with_fake(
        monkeypatch,
        mutated,
        status,
        splits,
    )

    pd.testing.assert_frame_equal(
        base_mat.x_train,
        new_mat.x_train,
    )
    pd.testing.assert_series_equal(
        base_mat.y_train,
        new_mat.y_train,
    )
    pd.testing.assert_frame_equal(
        base_mat.x_final_test,
        new_mat.x_final_test,
    )
    pd.testing.assert_frame_equal(
        base_pred.table,
        new_pred.table,
        check_exact=True,
    )


def test_mutating_final_test_status_outcome_flags_cannot_change_prediction_set(
    monkeypatch,
):
    dataset, status, splits = _synthetic_frames()

    base_mat, base_pred = _predict_with_fake(
        monkeypatch,
        dataset,
        status,
        splits,
    )

    mutated_status = status.copy()
    final_mask = splits["period"].eq("final_test")

    # Simulate arbitrary changes in outcome-derived supervised eligibility.
    mutated_status.loc[
        final_mask,
        "supervised_eligible",
    ] = ~mutated_status.loc[
        final_mask,
        "supervised_eligible",
    ]

    new_mat, new_pred = _predict_with_fake(
        monkeypatch,
        dataset,
        mutated_status,
        splits,
    )

    pd.testing.assert_frame_equal(
        base_mat.x_final_test,
        new_mat.x_final_test,
    )
    pd.testing.assert_frame_equal(
        base_pred.table,
        new_pred.table,
        check_exact=True,
    )


def test_mutating_post2022_features_cannot_change_training_sample():
    dataset, status, splits = _synthetic_frames()

    base = materialize_phase8_data(
        dataset,
        status,
        splits,
    )

    mutated = dataset.copy()
    final_mask = splits["period"].eq("final_test")
    mutated.loc[
        final_mask,
        list(PHASE8_FEATURES),
    ] += 10000.0

    new = materialize_phase8_data(
        mutated,
        status,
        splits,
    )

    pd.testing.assert_frame_equal(
        base.x_train,
        new.x_train,
    )
    pd.testing.assert_series_equal(
        base.y_train,
        new.y_train,
    )


def test_mutating_pre2022_targets_changes_training_only_not_contract():
    dataset, status, splits = _synthetic_frames()

    base = materialize_phase8_data(
        dataset,
        status,
        splits,
    )

    mutated = dataset.copy()
    train_index = base.train_index
    mutated.loc[
        train_index[0],
        "target",
    ] = 1.0 - float(
        mutated.loc[
            train_index[0],
            "target",
        ]
    )

    new = materialize_phase8_data(
        mutated,
        status,
        splits,
    )

    pd.testing.assert_frame_equal(
        base.x_train,
        new.x_train,
    )
    assert not base.y_train.equals(new.y_train)

    assert PHASE8_MODEL_CONFIG_ID == "lightgbm_lr0.1_leaves127"
    assert PHASE8_OPERATIONAL_THRESHOLD == 0.10


def test_phase8_prediction_api_has_no_target_argument():
    import inspect

    signature = inspect.signature(
        generate_phase8_predictions
    )

    assert tuple(signature.parameters) == (
        "materialized",
        "progress",
    )


def test_phase8_materialization_api_has_no_threshold_or_model_argument():
    import inspect

    signature = inspect.signature(
        materialize_phase8_data
    )

    assert tuple(signature.parameters) == (
        "dataset",
        "status",
        "splits",
    )


def test_phase8_modules_do_not_expose_selection_or_scoring_symbols():
    forbidden = {
        "optimize_phase6_threshold",
        "DEFAULT_THRESHOLD_GRID",
        "identify_alerts",
        "identify_events",
        "event_recall",
        "far_per_day",
        "score_final_test",
        "evaluate_final_test",
    }

    materialization_module = __import__(
        "src.final_test.materialization",
        fromlist=["*"],
    )
    prediction_module = __import__(
        "src.final_test.prediction",
        fromlist=["*"],
    )

    for name in forbidden:
        assert not hasattr(
            materialization_module,
            name,
        )
        assert not hasattr(
            prediction_module,
            name,
        )


def test_synthetic_end_to_end_dry_run_is_probability_only(monkeypatch):
    dataset, status, splits = _synthetic_frames()

    materialized, predictions = _predict_with_fake(
        monkeypatch,
        dataset,
        status,
        splits,
    )

    assert not materialized.x_train.empty
    assert not materialized.x_final_test.empty

    assert tuple(predictions.table.columns) == (
        "probability",
    )
    assert predictions.config_id == PHASE8_MODEL_CONFIG_ID
    assert (
        predictions.operational_threshold
        == PHASE8_OPERATIONAL_THRESHOLD
    )

    assert (
        predictions.table["probability"]
        .between(0.0, 1.0)
        .all()
    )

    for forbidden in (
        "target",
        "storm_id",
        "alert",
        "classification",
        "event_recall",
        "far_per_day",
    ):
        assert forbidden not in predictions.table.columns


def test_prediction_changes_when_final_test_features_change_but_not_targets(
    monkeypatch,
):
    dataset, status, splits = _synthetic_frames()

    _, base = _predict_with_fake(
        monkeypatch,
        dataset,
        status,
        splits,
    )

    feature_mutated = dataset.copy()
    final_mask = splits["period"].eq("final_test")
    feature_mutated.loc[
        final_mask,
        PHASE8_FEATURES[0],
    ] += 50.0

    _, changed = _predict_with_fake(
        monkeypatch,
        feature_mutated,
        status,
        splits,
    )

    assert not base.table.equals(
        changed.table
    )


def test_final_test_target_column_can_be_completely_missing_from_prediction_copy(
    monkeypatch,
):
    dataset, status, splits = _synthetic_frames()

    # Materialization still needs target for pre-2022 training. Demonstrate
    # that all protected targets may be unknown without changing prediction.
    final_mask = splits["period"].eq("final_test")
    dataset.loc[final_mask, "target"] = np.nan

    materialized, predictions = _predict_with_fake(
        monkeypatch,
        dataset,
        status,
        splits,
    )

    assert len(predictions.table) == len(
        materialized.x_final_test
    )
    assert tuple(predictions.table.columns) == (
        "probability",
    )
