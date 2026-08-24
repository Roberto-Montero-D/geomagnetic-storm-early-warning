"""Phase 4 walk-forward confirmation for frozen advancing strategies."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.baselines.extratrees import make_extratrees_model
from src.baselines.framework import DevelopmentFold, get_development_xy
from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST, PERIOD_INITIAL_TRAIN, PERIOD_VALIDATION_1,
    PERIOD_VALIDATION_2, PERIOD_VALIDATION_3,
)
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import DEFAULT_THRESHOLD_GRID
from src.feature_screening.screening import _events_for_validation
from .contract import (
    PHASE4_CONFIRMATION_FOLDS, PHASE4_EXPERIMENT_NAMES,
    PHASE4_EXTRATREES_PARAMS, PHASE4_FEATURES, PHASE4_MAX_FAR_PER_DAY,
)
from .screening import _threshold_curve
from .strategies import prepare_training_data
from .contract import PHASE4_EXPERIMENTS

PHASE4_ADVANCING_EXPERIMENTS = (
    "undersample_10_to_1",
    "none",
    "class_weight_1",
)

@dataclass(frozen=True)
class ConfirmationFoldResult:
    fold: str
    experiment: str
    threshold: float | None
    event_recall: float
    false_alarm_rate_per_day: float
    pr_auc: float
    operationally_feasible: bool
    threshold_table: pd.DataFrame

@dataclass(frozen=True)
class Phase4ConfirmationResult:
    folds: dict[tuple[str,str], ConfirmationFoldResult]
    ranking: pd.DataFrame
    selected_experiment: str | None

def _exp(name):
    return next(x for x in PHASE4_EXPERIMENTS if x.name==name)

def _validate_confirmation_fold(
    fold: DevelopmentFold,
    splits: pd.DataFrame,
    fold_name: str,
) -> None:
    expected = {
        "walk_forward_1": (
            (PERIOD_INITIAL_TRAIN, PERIOD_VALIDATION_1),
            (PERIOD_VALIDATION_2,),
        ),
        "walk_forward_2": (
            (PERIOD_INITIAL_TRAIN, PERIOD_VALIDATION_1, PERIOD_VALIDATION_2),
            (PERIOD_VALIDATION_3,),
        ),
    }

    if fold_name not in expected:
        raise ValueError(f"unknown Phase 4 confirmation fold: {fold_name}")

    for role, index in (
        ("train", fold.train_index),
        ("validation", fold.validation_index),
    ):
        if not index.isin(splits.index).all():
            raise ValueError(
                f"Phase 4 confirmation {role} contains timestamps absent from splits."
            )

        periods = splits.loc[index, "period"]
        if periods.eq(PERIOD_FINAL_TEST).any():
            raise ValueError(
                f"Phase 4 confirmation {role} touches the protected Final Test."
            )

    train_periods, validation_periods = expected[fold_name]

    expected_train_index = splits.index[
        splits["period"].isin(train_periods)
    ]
    expected_validation_index = splits.index[
        splits["period"].isin(validation_periods)
    ]

    if not fold.train_index.equals(expected_train_index):
        raise ValueError(
            f"{fold_name} training rows do not match the frozen temporal contract."
        )

    if not fold.validation_index.equals(expected_validation_index):
        raise ValueError(
            f"{fold_name} validation rows do not match the frozen temporal contract."
        )

    if len(fold.train_index.intersection(fold.validation_index)) != 0:
        raise ValueError(
            "Phase 4 confirmation train and validation overlap."
        )

    if (
        len(fold.train_index) > 0
        and len(fold.validation_index) > 0
        and fold.train_index.max() >= fold.validation_index.min()
    ):
        raise ValueError(
            "Phase 4 confirmation violates chronological order."
        )

def evaluate_confirmation_fold(dataset,fold,events,splits,fold_name,experiment,
                               *,thresholds=DEFAULT_THRESHOLD_GRID):
    if experiment not in PHASE4_ADVANCING_EXPERIMENTS:
        raise ValueError("Phase 4 confirmation accepts only frozen advancing experiments.")
    _validate_confirmation_fold(fold,splits,fold_name)
    xtr,ytr,xv,yv=get_development_xy(dataset,fold,PHASE4_FEATURES)
    prepared=prepare_training_data(xtr,ytr.astype(int),_exp(experiment))
    p=dict(PHASE4_EXTRATREES_PARAMS)
    model=make_extratrees_model(n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],random_state=p["random_state"])
    if prepared.class_weight is not None:
        model.set_params(class_weight=prepared.class_weight)
    model.fit(prepared.x,prepared.y.astype(int))
    prob=pd.Series(model.predict_proba(xv)[:,1],index=xv.index,dtype=float)
    scoped=_events_for_validation(events,prob.index)
    pr=float(average_precision_score(yv.astype(int),prob))
    tau,table=_threshold_curve(prob,scoped,thresholds,PHASE4_MAX_FAR_PER_DAY,3,6)
    if tau is None:
        return ConfirmationFoldResult(fold_name,experiment,None,np.nan,np.nan,pr,False,table)
    _,m=evaluate_operational_series(prob,scoped,threshold=tau,cooldown_hours=3,horizon_hours=6)
    return ConfirmationFoldResult(fold_name,experiment,tau,m.event_recall,
        m.false_alarm_rate_per_day,pr,True,table)

def rank_confirmation_candidates(results):
    rows=[]
    for order,name in enumerate(PHASE4_ADVANCING_EXPERIMENTS):
        rr=[results[(fold,name)] for fold in PHASE4_CONFIRMATION_FOLDS]
        feasible=all(x.operationally_feasible for x in rr)
        recalls=[x.event_recall for x in rr]
        prs=[x.pr_auc for x in rr]
        fars=[x.false_alarm_rate_per_day for x in rr]
        rows.append({
            "experiment":name,
            "confirmation_feasible":feasible,
            "minimum_event_recall":min(recalls) if feasible else np.nan,
            "mean_event_recall":float(np.mean(recalls)) if feasible else np.nan,
            "mean_pr_auc":float(np.mean(prs)),
            "mean_false_alarm_rate_per_day":float(np.mean(fars)) if feasible else np.nan,
            "_order":order,
        })
    ranking=pd.DataFrame(rows).sort_values(
        ["confirmation_feasible","minimum_event_recall","mean_event_recall",
         "mean_pr_auc","mean_false_alarm_rate_per_day","_order"],
        ascending=[False,False,False,False,True,True],
        na_position="last",kind="mergesort").reset_index(drop=True)
    feasible=ranking.loc[ranking.confirmation_feasible,"experiment"]
    selected=None if feasible.empty else str(feasible.iloc[0])
    return ranking.drop(columns="_order"),selected

def evaluate_phase4_confirmation(dataset,folds,events,splits,*,thresholds=DEFAULT_THRESHOLD_GRID):
    results={}
    for fold_name in PHASE4_CONFIRMATION_FOLDS:
        if fold_name not in folds:
            raise ValueError(f"missing required confirmation fold: {fold_name}")
        for name in PHASE4_ADVANCING_EXPERIMENTS:
            results[(fold_name,name)]=evaluate_confirmation_fold(
                dataset,folds[fold_name],events,splits,fold_name,name,thresholds=thresholds)
    ranking,selected=rank_confirmation_candidates(results)
    return Phase4ConfirmationResult(results,ranking,selected)
