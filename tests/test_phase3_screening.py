import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import assign_temporal_periods
from src.feature_screening.manifests import PHASE3_FEATURE_SETS
from src.feature_screening.screening import (
    ScreeningExperimentResult,
    evaluate_screening_experiment,
    rank_screening_experiments,
)


def _dataset():
    train=pd.date_range(
        "2016-12-29 00:00",periods=72,freq="h",name="prediction_time"
    )
    valid=pd.date_range(
        "2017-01-01 00:00",periods=48,freq="h",name="prediction_time"
    )
    index=train.append(valid)
    rng=np.random.default_rng(123)
    columns=PHASE3_FEATURE_SETS["E"]
    frame=pd.DataFrame(
        rng.normal(size=(len(index),len(columns))),
        index=index,columns=columns,
    )
    frame["target"]=([0,1]*60)
    return frame,DevelopmentFold("screening",train,valid)


def _events():
    return pd.DataFrame({
        "event_id":[1],
        "start_time":[pd.Timestamp("2017-01-02 00:00")],
        "end_time":[pd.Timestamp("2017-01-02 03:00")],
        "boundary_status":["complete"],
    })


def test_single_experiment_returns_probability_and_pr_auc():
    dataset,fold=_dataset()
    splits=assign_temporal_periods(dataset.index)
    result=evaluate_screening_experiment(
        dataset,fold,_events(),splits,"A",
        thresholds=[0.2,0.5,0.8],
    )
    assert result.experiment=="A"
    assert result.n_features==len(PHASE3_FEATURE_SETS["A"])
    pd.testing.assert_index_equal(
        result.validation_probability.index,fold.validation_index
    )
    assert 0.0 <= result.pr_auc <= 1.0


def test_screening_rejects_final_test_validation():
    dataset,fold=_dataset()
    final=pd.date_range(
        "2022-01-01",periods=4,freq="h",name="prediction_time"
    )
    extra=dataset.iloc[:4].copy()
    extra.index=final
    augmented=pd.concat([dataset,extra])
    splits=assign_temporal_periods(augmented.index)
    bad=DevelopmentFold("screening",fold.train_index,final)
    with pytest.raises(ValueError, match="protected Final Test"):
        evaluate_screening_experiment(
            augmented,bad,_events(),splits,"A",thresholds=[0.5]
        )


def test_screening_rejects_later_validation_period():
    dataset,fold=_dataset()
    later=pd.date_range(
        "2019-01-01",periods=4,freq="h",name="prediction_time"
    )
    extra=dataset.iloc[:4].copy()
    extra.index=later
    augmented=pd.concat([dataset,extra])
    splits=assign_temporal_periods(augmented.index)
    bad=DevelopmentFold("screening",fold.train_index,later)
    with pytest.raises(ValueError, match="validation_1"):
        evaluate_screening_experiment(
            augmented,bad,_events(),splits,"A",thresholds=[0.5]
        )


def _fake(name,recall,pr,far,feasible=True):
    return ScreeningExperimentResult(
        experiment=name,
        n_features=len(PHASE3_FEATURE_SETS[name]),
        threshold=0.5 if feasible else None,
        event_recall=recall if feasible else np.nan,
        false_alarm_rate_per_day=far if feasible else np.nan,
        pr_auc=pr,
        operationally_feasible=feasible,
        validation_probability=pd.Series(dtype=float),
        threshold_table=pd.DataFrame(),
    )


def test_ranking_uses_frozen_priority_and_advances_three():
    experiments={
        "A":_fake("A",0.7,0.2,0.15),
        "B":_fake("B",0.8,0.1,0.15),
        "C":_fake("C",0.8,0.3,0.19),
        "D":_fake("D",0.8,0.3,0.10),
        "E":_fake("E",0.1,0.9,0.01),
    }
    ranking,advancing=rank_screening_experiments(experiments)
    assert ranking["experiment"].tolist()[:3]==["D","C","B"]
    assert advancing==("D","C","B")


def test_smaller_set_breaks_complete_tie():
    experiments={
        name:_fake(name,0.5,0.2,0.1)
        for name in ("A","B","C","D","E")
    }
    _,advancing=rank_screening_experiments(experiments)
    assert advancing==("A","B","C")


def test_all_feasible_sets_advance_when_fewer_than_three():
    experiments={
        "A":_fake("A",0.5,0.2,0.1),
        "B":_fake("B",0.4,0.2,0.1),
        "C":_fake("C",0,0.2,0,False),
        "D":_fake("D",0,0.2,0,False),
        "E":_fake("E",0,0.2,0,False),
    }
    _,advancing=rank_screening_experiments(experiments)
    assert advancing==("A","B")
