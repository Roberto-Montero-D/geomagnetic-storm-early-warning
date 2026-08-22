import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import assign_temporal_periods
from src.feature_screening.manifests import PHASE3_FEATURE_SETS
from src.feature_screening.screening import evaluate_screening_experiment


def _dataset():
    train=pd.date_range(
        "2016-12-29 00:00",periods=72,freq="h",name="prediction_time"
    )
    valid=pd.date_range(
        "2017-01-01 00:00",periods=48,freq="h",name="prediction_time"
    )
    index=train.append(valid)
    rng=np.random.default_rng(77)
    frame=pd.DataFrame(
        rng.normal(size=(len(index),len(PHASE3_FEATURE_SETS["E"]))),
        index=index,
        columns=PHASE3_FEATURE_SETS["E"],
    )
    frame["target"]=[0,1]*60
    return frame,DevelopmentFold("screening",train,valid)


def _events():
    return pd.DataFrame({
        "event_id":[1],
        "start_time":[pd.Timestamp("2017-01-02 00:00")],
        "end_time":[pd.Timestamp("2017-01-02 03:00")],
        "boundary_status":["complete"],
    })


def _run(dataset,fold,events=None):
    splits=assign_temporal_periods(dataset.index)
    return evaluate_screening_experiment(
        dataset,
        fold,
        _events() if events is None else events,
        splits,
        "A",
        thresholds=[0.25,0.5,0.75],
    )


def test_validation_target_mutation_does_not_change_predictions():
    dataset,fold=_dataset()
    before=_run(dataset,fold)

    mutated=dataset.copy()
    mutated.loc[fold.validation_index,"target"] = (
        1-mutated.loc[fold.validation_index,"target"]
    )
    after=_run(mutated,fold)

    pd.testing.assert_series_equal(
        before.validation_probability,
        after.validation_probability,
    )


def test_row_outside_screening_fold_cannot_change_predictions():
    dataset,fold=_dataset()
    before=_run(dataset,fold)

    extra=dataset.iloc[[0]].copy()
    extra.index=pd.DatetimeIndex(
        [pd.Timestamp("2021-06-01 00:00")],
        name="prediction_time",
    )
    extra.loc[:,list(PHASE3_FEATURE_SETS["E"])] = 1e9
    extra["target"]=1
    augmented=pd.concat([dataset,extra])

    after=_run(augmented,fold)

    pd.testing.assert_series_equal(
        before.validation_probability,
        after.validation_probability,
        check_freq=False,
    )


def test_future_event_outside_validation_cannot_change_metrics():
    dataset,fold=_dataset()
    before=_run(dataset,fold)

    events=pd.concat([
        _events(),
        pd.DataFrame({
            "event_id":[999],
            "start_time":[pd.Timestamp("2021-06-01 00:00")],
            "end_time":[pd.Timestamp("2021-06-01 03:00")],
            "boundary_status":["complete"],
        }),
    ],ignore_index=True)
    after=_run(dataset,fold,events)

    assert before.threshold == after.threshold
    assert before.event_recall == after.event_recall
    assert before.false_alarm_rate_per_day == after.false_alarm_rate_per_day
    assert before.pr_auc == after.pr_auc


def test_validation_2_row_is_rejected_even_if_manually_supplied():
    dataset,fold=_dataset()
    later=pd.date_range(
        "2019-01-01",periods=2,freq="h",name="prediction_time"
    )
    extra=dataset.iloc[:2].copy()
    extra.index=later
    augmented=pd.concat([dataset,extra])
    splits=assign_temporal_periods(augmented.index)

    bad=DevelopmentFold(
        "screening",
        fold.train_index,
        fold.validation_index.append(later),
    )
    with pytest.raises(ValueError,match="validation_1"):
        evaluate_screening_experiment(
            augmented,bad,_events(),splits,"A",thresholds=[0.5]
        )


def test_validation_1_row_cannot_enter_training():
    dataset,fold=_dataset()
    bad=DevelopmentFold(
        "screening",
        fold.train_index.append(fold.validation_index[:1]),
        fold.validation_index[1:],
    )
    splits=assign_temporal_periods(dataset.index)

    with pytest.raises(ValueError,match="initial_train"):
        evaluate_screening_experiment(
            dataset,bad,_events(),splits,"A",thresholds=[0.5]
        )


def test_invalid_far_limit_is_rejected():
    dataset,fold=_dataset()
    splits=assign_temporal_periods(dataset.index)

    with pytest.raises(ValueError,match="non-negative"):
        evaluate_screening_experiment(
            dataset,fold,_events(),splits,"A",
            thresholds=[0.5],
            max_far_per_day=-0.1,
        )
