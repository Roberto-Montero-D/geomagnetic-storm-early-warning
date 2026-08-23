import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import assign_temporal_periods
from src.imbalance.contract import PHASE4_FEATURES
from src.imbalance.screening import (
    _fit_experiment,
    _validate_phase4_screening_fold,
    evaluate_imbalance_experiment,
)


def _dataset():
    train = pd.date_range(
        "2016-12-24 00:00",
        periods=192,
        freq="h",
        name="prediction_time",
    )
    validation = pd.date_range(
        "2017-01-01 00:00",
        periods=48,
        freq="h",
        name="prediction_time",
    )
    index = train.append(validation)

    rng = np.random.default_rng(987)
    frame = pd.DataFrame(
        rng.normal(size=(len(index), len(PHASE4_FEATURES))),
        index=index,
        columns=PHASE4_FEATURES,
    )

    # Training prevalence is 12/192 = 6.25%, sufficiently imbalanced for all
    # frozen undersampling ratios and with enough positives for SMOTE k=7.
    target = np.zeros(len(index), dtype=int)
    train_positive_positions = np.linspace(4, 187, 12, dtype=int)
    target[train_positive_positions] = 1

    # Validation has both classes for PR-AUC.
    target[len(train) + np.array([6, 18, 30, 42])] = 1
    frame["target"] = target

    fold = DevelopmentFold(
        "screening",
        train,
        validation,
    )
    return frame, fold


def _events():
    return pd.DataFrame(
        {
            "event_id": [1],
            "start_time": [pd.Timestamp("2017-01-02 06:00")],
            "end_time": [pd.Timestamp("2017-01-02 09:00")],
            "boundary_status": ["complete"],
        }
    )


def test_valid_phase4_screening_period_contract_passes():
    dataset, fold = _dataset()
    splits = assign_temporal_periods(dataset.index)
    _validate_phase4_screening_fold(fold, splits)


def test_validation_2_row_is_rejected():
    dataset, fold = _dataset()

    later = pd.date_range(
        "2019-01-01",
        periods=1,
        freq="h",
        name="prediction_time",
    )
    extra = dataset.iloc[[0]].copy()
    extra.index = later
    augmented = pd.concat([dataset, extra])
    splits = assign_temporal_periods(augmented.index)

    bad = DevelopmentFold(
        "screening",
        fold.train_index,
        fold.validation_index.append(later),
    )

    with pytest.raises(ValueError, match="validation_1"):
        _validate_phase4_screening_fold(bad, splits)


def test_validation_1_row_cannot_enter_training():
    dataset, fold = _dataset()
    splits = assign_temporal_periods(dataset.index)

    bad = DevelopmentFold(
        "screening",
        fold.train_index.append(fold.validation_index[:1]),
        fold.validation_index[1:],
    )

    with pytest.raises(ValueError, match="initial_train"):
        _validate_phase4_screening_fold(bad, splits)


def test_final_test_row_is_rejected_explicitly():
    dataset, fold = _dataset()

    final = pd.date_range(
        "2022-01-01",
        periods=1,
        freq="h",
        name="prediction_time",
    )
    extra = dataset.iloc[[0]].copy()
    extra.index = final
    augmented = pd.concat([dataset, extra])
    splits = assign_temporal_periods(augmented.index)

    bad = DevelopmentFold(
        "screening",
        fold.train_index,
        fold.validation_index.append(final),
    )

    with pytest.raises(ValueError, match="protected Final Test"):
        _validate_phase4_screening_fold(bad, splits)


def test_timestamp_absent_from_split_table_is_rejected():
    dataset, fold = _dataset()
    splits = assign_temporal_periods(dataset.index)

    unknown = pd.DatetimeIndex(
        [pd.Timestamp("2017-02-01 00:00")],
        name="prediction_time",
    )
    bad = DevelopmentFold(
        "screening",
        fold.train_index,
        fold.validation_index.append(unknown),
    )

    with pytest.raises(ValueError, match="absent from splits"):
        _validate_phase4_screening_fold(bad, splits)


@pytest.mark.parametrize(
    "experiment",
    [
        "none",
        "class_weight_10",
        "undersample_5_to_1",
        "smote_k5",
        "borderline_smote_k5",
        "smote_enn",
    ],
)
def test_every_strategy_predicts_on_exact_untouched_validation_index(experiment):
    dataset, fold = _dataset()
    probability, y_validation = _fit_experiment(
        dataset,
        fold,
        experiment,
    )

    assert probability.index.equals(fold.validation_index)
    assert y_validation.index.equals(fold.validation_index)
    assert len(probability) == len(fold.validation_index)


@pytest.mark.parametrize(
    "experiment",
    [
        "none",
        "class_weight_10",
        "undersample_5_to_1",
        "smote_k5",
        "borderline_smote_k5",
        "smote_enn",
    ],
)
def test_validation_target_mutation_cannot_change_probabilities(experiment):
    dataset, fold = _dataset()

    before, _ = _fit_experiment(dataset, fold, experiment)

    mutated = dataset.copy()
    mutated.loc[fold.validation_index, "target"] = (
        1 - mutated.loc[fold.validation_index, "target"]
    )

    after, _ = _fit_experiment(mutated, fold, experiment)

    pd.testing.assert_series_equal(before, after)


def test_row_outside_fold_cannot_change_resampled_model_predictions():
    dataset, fold = _dataset()
    splits = assign_temporal_periods(dataset.index)

    before = evaluate_imbalance_experiment(
        dataset,
        fold,
        _events(),
        splits,
        "smote_k5",
        thresholds=[0.25, 0.5, 0.75],
    )

    extra_time = pd.Timestamp("2018-06-01 00:00")
    extra = dataset.iloc[[0]].copy()
    extra.index = pd.DatetimeIndex(
        [extra_time],
        name="prediction_time",
    )
    extra.loc[:, list(PHASE4_FEATURES)] = 1e9
    extra["target"] = 1

    augmented = pd.concat([dataset, extra])
    augmented_splits = assign_temporal_periods(augmented.index)

    after = evaluate_imbalance_experiment(
        augmented,
        fold,
        _events(),
        augmented_splits,
        "smote_k5",
        thresholds=[0.25, 0.5, 0.75],
    )
    pd.testing.assert_series_equal(
        before.validation_probability,
        after.validation_probability,
        check_freq=False,
    )
    pd.testing.assert_frame_equal(
        before.threshold_table,
        after.threshold_table,
    )
    assert before.threshold == after.threshold
    assert before.event_recall == after.event_recall
    assert before.false_alarm_rate_per_day == after.false_alarm_rate_per_day
    assert before.pr_auc == after.pr_auc


def test_future_event_outside_validation_cannot_change_metrics():
    dataset, fold = _dataset()
    splits = assign_temporal_periods(dataset.index)

    before = evaluate_imbalance_experiment(
        dataset,
        fold,
        _events(),
        splits,
        "none",
        thresholds=[0.25, 0.5, 0.75],
    )

    future = pd.concat(
        [
            _events(),
            pd.DataFrame(
                {
                    "event_id": [999],
                    "start_time": [pd.Timestamp("2019-01-01 00:00")],
                    "end_time": [pd.Timestamp("2019-01-01 03:00")],
                    "boundary_status": ["complete"],
                }
            ),
        ],
        ignore_index=True,
    )

    after = evaluate_imbalance_experiment(
        dataset,
        fold,
        future,
        splits,
        "none",
        thresholds=[0.25, 0.5, 0.75],
    )

    pd.testing.assert_frame_equal(
        before.threshold_table,
        after.threshold_table,
    )
    assert before.threshold == after.threshold
    assert before.event_recall == after.event_recall
    assert before.false_alarm_rate_per_day == after.false_alarm_rate_per_day
    assert before.pr_auc == after.pr_auc
