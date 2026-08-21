import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import (
    DEVELOPMENT_FOLD_NAMES,
    build_development_folds,
    get_development_xy,
)
from src.dataset.row_status import build_row_status
from src.dataset.temporal_splits import assign_temporal_periods
from src.features.integrated import PRIMARY_FEATURE_COLUMNS


def _dataset():
    times = pd.DatetimeIndex(
        [
            "2016-12-31 22:00",
            "2016-12-31 23:00",
            "2017-01-01 00:00",
            "2018-12-31 23:00",
            "2019-01-01 00:00",
            "2020-12-31 23:00",
            "2021-01-01 00:00",
            "2021-12-31 23:00",
            "2022-01-01 00:00",
            "2022-01-01 01:00",
        ],
        name="prediction_time",
    )
    frame = pd.DataFrame(
        1.0,
        index=times,
        columns=list(PRIMARY_FEATURE_COLUMNS),
    )
    frame["target"] = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    return frame


def _components(dataset=None):
    if dataset is None:
        dataset = _dataset()
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)
    return dataset, status, splits


def test_framework_returns_exact_three_frozen_development_folds():
    dataset, status, splits = _components()

    folds = build_development_folds(dataset, status, splits)

    assert tuple(folds) == DEVELOPMENT_FOLD_NAMES


def test_final_test_is_absent_from_every_materialized_fold():
    dataset, status, splits = _components()
    folds = build_development_folds(dataset, status, splits)
    final_index = dataset.index[splits["is_final_test"]]

    for fold in folds.values():
        assert not fold.train_index.isin(final_index).any()
        assert not fold.validation_index.isin(final_index).any()


def test_ineligible_rows_are_removed_only_from_materialized_supervised_folds():
    dataset = _dataset()
    missing_time = pd.Timestamp("2019-01-01 00:00")
    dataset.loc[
        missing_time,
        PRIMARY_FEATURE_COLUMNS[0],
    ] = np.nan

    dataset, status, splits = _components(dataset)
    folds = build_development_folds(dataset, status, splits)

    assert missing_time in dataset.index
    assert not status.loc[missing_time, "supervised_eligible"]
    assert missing_time not in folds["walk_forward_1"].validation_index


def test_each_fold_is_strictly_chronological():
    dataset, status, splits = _components()
    folds = build_development_folds(dataset, status, splits)

    for fold in folds.values():
        assert fold.train_index.max() < fold.validation_index.min()


def test_expanding_training_membership_matches_phase1_contract():
    dataset, status, splits = _components()
    folds = build_development_folds(dataset, status, splits)

    assert pd.Timestamp("2017-01-01 00:00") not in folds["screening"].train_index
    assert pd.Timestamp("2017-01-01 00:00") in folds["walk_forward_1"].train_index
    assert pd.Timestamp("2019-01-01 00:00") in folds["walk_forward_2"].train_index
    assert pd.Timestamp("2021-01-01 00:00") in folds["walk_forward_2"].validation_index


def test_get_development_xy_exposes_only_requested_features():
    dataset, status, splits = _components()
    fold = build_development_folds(dataset, status, splits)["screening"]
    features = list(PRIMARY_FEATURE_COLUMNS[:3])

    x_train, y_train, x_val, y_val = get_development_xy(
        dataset, fold, features
    )

    assert x_train.columns.tolist() == features
    assert x_val.columns.tolist() == features
    assert "target" not in x_train.columns
    assert "target" not in x_val.columns
    assert y_train.name == "target"
    assert y_val.name == "target"


def test_xy_indices_match_materialized_fold_indices():
    dataset, status, splits = _components()
    fold = build_development_folds(dataset, status, splits)["walk_forward_2"]

    x_train, y_train, x_val, y_val = get_development_xy(
        dataset, fold, list(PRIMARY_FEATURE_COLUMNS[:2])
    )

    pd.testing.assert_index_equal(x_train.index, fold.train_index)
    pd.testing.assert_index_equal(y_train.index, fold.train_index)
    pd.testing.assert_index_equal(x_val.index, fold.validation_index)
    pd.testing.assert_index_equal(y_val.index, fold.validation_index)


def test_target_cannot_be_requested_as_predictor():
    dataset, status, splits = _components()
    fold = build_development_folds(dataset, status, splits)["screening"]

    with pytest.raises(ValueError, match="target"):
        get_development_xy(dataset, fold, ["target"])


def test_empty_or_duplicate_feature_manifest_raises():
    dataset, status, splits = _components()
    fold = build_development_folds(dataset, status, splits)["screening"]

    with pytest.raises(ValueError, match="must not be empty"):
        get_development_xy(dataset, fold, [])

    feature = PRIMARY_FEATURE_COLUMNS[0]
    with pytest.raises(ValueError, match="unique"):
        get_development_xy(dataset, fold, [feature, feature])


def test_missing_requested_feature_raises():
    dataset, status, splits = _components()
    fold = build_development_folds(dataset, status, splits)["screening"]

    with pytest.raises(ValueError, match="missing requested"):
        get_development_xy(dataset, fold, ["does_not_exist"])


def test_misaligned_status_raises():
    dataset, status, splits = _components()
    status = status.iloc[:-1]

    with pytest.raises(ValueError, match="indices must match"):
        build_development_folds(dataset, status, splits)


def test_misaligned_splits_raises():
    dataset, status, splits = _components()
    splits = splits.iloc[:-1]

    with pytest.raises(ValueError, match="indices must match"):
        build_development_folds(dataset, status, splits)


def test_unknown_target_cannot_survive_into_xy():
    dataset, status, splits = _components()
    folds = build_development_folds(dataset, status, splits)
    fold = folds["screening"]

    corrupted = dataset.copy()
    corrupted.loc[fold.train_index[0], "target"] = np.nan

    with pytest.raises(AssertionError, match="unknown target"):
        get_development_xy(
            corrupted,
            fold,
            list(PRIMARY_FEATURE_COLUMNS[:2]),
        )
