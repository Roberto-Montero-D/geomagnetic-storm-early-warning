import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import assign_temporal_periods
from src.evaluation.cross_fold import (
    DETERMINISTIC_THRESHOLD,
    _events_for_validation,
    evaluate_development_folds,
)
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS


def _dataset_and_folds():
    idx1 = pd.date_range(
        "2017-01-01", periods=30, freq="h", name="prediction_time"
    )
    idx2 = pd.date_range(
        "2019-01-01", periods=30, freq="h", name="prediction_time"
    )
    idx3 = pd.date_range(
        "2021-01-01", periods=30, freq="h", name="prediction_time"
    )
    index = idx1.append(idx2).append(idx3)

    rng = np.random.default_rng(9)
    dataset = pd.DataFrame(
        rng.normal(size=(90, len(PRIMARY_RAW_FEATURE_COLUMNS))),
        index=index,
        columns=list(PRIMARY_RAW_FEATURE_COLUMNS),
    )
    dataset["bz_gsm"] = rng.normal(0, 5, 90)
    dataset["bt"] = np.abs(rng.normal(7, 2, 90))
    dataset["speed"] = rng.normal(500, 60, 90)
    dataset["density"] = np.abs(rng.normal(6, 2, 90))
    dataset["flow_pressure"] = np.abs(rng.normal(2, 0.5, 90))

    for column in [
        "kp_lag_1h",
        "kp_lag_3h",
        "kp_lag_6h",
        "kp_lag_12h",
        "kp_lag_24h",
    ]:
        dataset[column] = rng.uniform(0, 8, 90)

    dataset["target"] = [0, 1] * 45

    folds = [
        DevelopmentFold("f1", idx1[:20], idx1[20:]),
        DevelopmentFold(
            "f2",
            idx1.append(idx2[:20]),
            idx2[20:],
        ),
        DevelopmentFold(
            "f3",
            idx1.append(idx2).append(idx3[:20]),
            idx3[20:],
        ),
    ]
    splits = assign_temporal_periods(index)
    return dataset, folds, splits


def _events():
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "start_time": [
                pd.Timestamp("2017-01-02 04:00"),
                pd.Timestamp("2019-01-02 04:00"),
                pd.Timestamp("2021-01-02 04:00"),
            ],
            "end_time": [
                pd.Timestamp("2017-01-02 05:00"),
                pd.Timestamp("2019-01-02 05:00"),
                pd.Timestamp("2021-01-02 05:00"),
            ],
            "boundary_status": ["complete"] * 3,
        }
    )


def test_cross_fold_returns_four_by_three_metric_rows():
    dataset, folds, splits = _dataset_and_folds()
    result = evaluate_development_folds(
        dataset,
        folds,
        _events(),
        splits,
        thresholds=[0.25, 0.5, 0.75],
    )

    assert len(result.fold_metrics) == 12
    assert set(result.fold_metrics["baseline"]) == {
        "B0_persistence",
        "B1_physical",
        "B2_logistic",
        "B3_extratrees",
    }
    assert set(result.fold_metrics["fold"]) == {"f1", "f2", "f3"}


def test_deterministic_threshold_is_fixed_not_selected():
    dataset, folds, splits = _dataset_and_folds()
    result = evaluate_development_folds(
        dataset,
        folds,
        _events(),
        splits,
        thresholds=[0.25, 0.5, 0.75],
    )

    assert (
        result.selected_thresholds["B0_persistence"]
        == DETERMINISTIC_THRESHOLD
    )
    assert (
        result.selected_thresholds["B1_physical"]
        == DETERMINISTIC_THRESHOLD
    )
    assert "B0_persistence" not in result.threshold_tables
    assert "B1_physical" not in result.threshold_tables


def test_probabilistic_threshold_tables_preserve_requested_grid():
    dataset, folds, splits = _dataset_and_folds()
    result = evaluate_development_folds(
        dataset,
        folds,
        _events(),
        splits,
        thresholds=[0.25, 0.5, 0.75],
    )

    for name in ("B2_logistic", "B3_extratrees"):
        assert result.threshold_tables[name]["threshold"].tolist() == [
            0.25,
            0.5,
            0.75,
        ]


def test_fold_metrics_keep_validation_windows_separate():
    dataset, folds, splits = _dataset_and_folds()
    result = evaluate_development_folds(
        dataset,
        folds,
        _events(),
        splits,
        thresholds=[0.5],
    )

    assert (
        result.fold_metrics.groupby(["baseline", "fold"])
        .size()
        .eq(1)
        .all()
    )


def test_event_after_validation_end_is_not_evaluable():
    events = pd.DataFrame(
        {
            "event_id": [1],
            "start_time": [pd.Timestamp("2021-01-02 08:00")],
            "end_time": [pd.Timestamp("2021-01-02 10:00")],
            "boundary_status": ["complete"],
        }
    )
    index = pd.date_range(
        "2021-01-01 20:00",
        periods=10,
        freq="h",
    )

    scoped = _events_for_validation(events, index, horizon_hours=6)
    assert scoped.empty


def test_final_test_validation_fold_is_rejected():
    dataset, folds, splits = _dataset_and_folds()

    final_index = pd.date_range(
        "2022-01-01",
        periods=2,
        freq="h",
        name="prediction_time",
    )
    extra = dataset.iloc[:2].copy()
    extra.index = final_index

    augmented = pd.concat([dataset, extra])
    augmented_splits = assign_temporal_periods(augmented.index)

    bad_fold = DevelopmentFold(
        "bad_final_test",
        folds[0].train_index,
        final_index,
    )

    with pytest.raises(ValueError, match="protected Final Test"):
        evaluate_development_folds(
            augmented,
            [bad_fold],
            _events(),
            augmented_splits,
            thresholds=[0.5],
        )


def test_final_test_training_fold_is_rejected():
    dataset, folds, splits = _dataset_and_folds()

    final_index = pd.date_range(
        "2022-01-01",
        periods=2,
        freq="h",
        name="prediction_time",
    )
    extra = dataset.iloc[:2].copy()
    extra.index = final_index

    augmented = pd.concat([dataset, extra])
    augmented_splits = assign_temporal_periods(augmented.index)

    bad_train = folds[0].train_index.append(final_index)
    bad_fold = DevelopmentFold(
        "bad_final_test_train",
        bad_train,
        folds[0].validation_index,
    )

    with pytest.raises(ValueError, match="protected Final Test"):
        evaluate_development_folds(
            augmented,
            [bad_fold],
            _events(),
            augmented_splits,
            thresholds=[0.5],
        )


def test_unsorted_threshold_grid_is_rejected():
    dataset, folds, splits = _dataset_and_folds()

    with pytest.raises(ValueError, match="sorted"):
        evaluate_development_folds(
            dataset,
            folds,
            _events(),
            splits,
            thresholds=[0.75, 0.25, 0.5],
        )


def test_duplicate_threshold_grid_is_rejected():
    dataset, folds, splits = _dataset_and_folds()

    with pytest.raises(ValueError, match="unique"):
        evaluate_development_folds(
            dataset,
            folds,
            _events(),
            splits,
            thresholds=[0.5, 0.5],
        )


def test_negative_far_limit_is_rejected():
    dataset, folds, splits = _dataset_and_folds()

    with pytest.raises(ValueError, match="non-negative"):
        evaluate_development_folds(
            dataset,
            folds,
            _events(),
            splits,
            thresholds=[0.5],
            max_far_per_day=-0.1,
        )
