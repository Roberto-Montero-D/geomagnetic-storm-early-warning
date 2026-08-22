"""Frozen Phase 3 walk-forward confirmation for A/E/C."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.baselines.extratrees import make_extratrees_model
from src.baselines.framework import DevelopmentFold, get_development_xy
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import DEFAULT_MAX_FAR_PER_DAY, DEFAULT_THRESHOLD_GRID
from src.feature_screening.manifests import PHASE3_EXTRATREES_PARAMS, PHASE3_FEATURE_SETS
from src.feature_screening.screening import _events_for_validation, _select_single_fold_threshold

PHASE3_ADVANCING_EXPERIMENTS = ("A", "E", "C")
PHASE3_CONFIRMATION_FOLDS = ("walk_forward_1", "walk_forward_2")


@dataclass(frozen=True)
class ConfirmationFoldResult:
    experiment: str
    fold: str
    threshold: float | None
    event_recall: float
    false_alarm_rate_per_day: float
    pr_auc: float
    operationally_feasible: bool
    threshold_table: pd.DataFrame


@dataclass(frozen=True)
class Phase3ConfirmationResult:
    fold_results: dict[tuple[str, str], ConfirmationFoldResult]
    ranking: pd.DataFrame
    selected_experiment: str | None


def _fit_candidate(dataset, fold, experiment):
    features=PHASE3_FEATURE_SETS[experiment]
    x_train,y_train,x_val,y_val=get_development_xy(dataset,fold,features)
    if x_train.isna().any().any() or x_val.isna().any().any():
        raise AssertionError("Phase 3 confirmation predictors contain missing values.")
    model=make_extratrees_model(
        n_estimators=PHASE3_EXTRATREES_PARAMS["n_estimators"],
        max_depth=PHASE3_EXTRATREES_PARAMS["max_depth"],
        random_state=PHASE3_EXTRATREES_PARAMS["random_state"],
    )
    model.fit(x_train,y_train.astype(int))
    p=pd.Series(model.predict_proba(x_val)[:,1],index=x_val.index,name="probability")
    return p,y_val.astype(int)


def evaluate_confirmation_fold(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
    events: pd.DataFrame,
    experiment: str,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day: float=DEFAULT_MAX_FAR_PER_DAY,
    progress: bool=False,
) -> ConfirmationFoldResult:
    if experiment not in PHASE3_ADVANCING_EXPERIMENTS:
        raise ValueError("Only frozen Phase 3 candidates A, E, and C may be confirmed.")
    if fold.name not in PHASE3_CONFIRMATION_FOLDS:
        raise ValueError("Confirmation requires walk_forward_1 or walk_forward_2.")

    if progress:
        print(f"    {fold.name}: fitting {experiment} ({len(PHASE3_FEATURE_SETS[experiment])} features)...",flush=True)

    probability,y_val=_fit_candidate(dataset,fold,experiment)
    scoped_events=_events_for_validation(events,probability.index)
    pr_auc=float(average_precision_score(y_val.to_numpy(),probability.to_numpy()))
    threshold,table=_select_single_fold_threshold(
        probability,scoped_events,
        thresholds=thresholds,max_far_per_day=max_far_per_day,
        cooldown_hours=3,horizon_hours=6,
        progress=progress,experiment=f"{fold.name}/{experiment}",
    )
    if threshold is None:
        return ConfirmationFoldResult(experiment,fold.name,None,np.nan,np.nan,pr_auc,False,table)

    _,metrics=evaluate_operational_series(
        probability,scoped_events,threshold=threshold,
        cooldown_hours=3,horizon_hours=6,
    )
    return ConfirmationFoldResult(
        experiment,fold.name,threshold,metrics.event_recall,
        metrics.false_alarm_rate_per_day,pr_auc,True,table
    )


def rank_confirmation_candidates(fold_results):
    rows=[]
    order={name:i for i,name in enumerate(PHASE3_ADVANCING_EXPERIMENTS)}
    for experiment in PHASE3_ADVANCING_EXPERIMENTS:
        items=[fold_results[(experiment,f)] for f in PHASE3_CONFIRMATION_FOLDS]
        feasible=all(x.operationally_feasible for x in items)
        recalls=[x.event_recall for x in items]
        prs=[x.pr_auc for x in items]
        fars=[x.false_alarm_rate_per_day for x in items]
        rows.append({
            "experiment":experiment,
            "confirmation_feasible":feasible,
            "minimum_event_recall":min(recalls) if feasible else np.nan,
            "mean_event_recall":float(np.mean(recalls)) if feasible else np.nan,
            "mean_pr_auc":float(np.mean(prs)),
            "mean_false_alarm_rate_per_day":float(np.mean(fars)) if feasible else np.nan,
            "_order":order[experiment],
        })
    ranking=pd.DataFrame(rows).sort_values(
        ["confirmation_feasible","minimum_event_recall","mean_event_recall",
         "mean_pr_auc","mean_false_alarm_rate_per_day","_order"],
        ascending=[False,False,False,False,True,True],
        na_position="last",kind="mergesort"
    ).reset_index(drop=True)
    feasible=ranking.loc[ranking["confirmation_feasible"],"experiment"]
    selected=None if feasible.empty else str(feasible.iloc[0])
    return ranking.drop(columns="_order"),selected


def evaluate_phase3_confirmation(dataset,fold_map,events,*,progress=False):
    missing=[f for f in PHASE3_CONFIRMATION_FOLDS if f not in fold_map]
    if missing:
        raise ValueError(f"Missing confirmation folds: {missing}")
    results={}
    for fold_name in PHASE3_CONFIRMATION_FOLDS:
        for experiment in PHASE3_ADVANCING_EXPERIMENTS:
            results[(experiment,fold_name)]=evaluate_confirmation_fold(
                dataset,fold_map[fold_name],events,experiment,progress=progress
            )
    ranking,selected=rank_confirmation_candidates(results)
    return Phase3ConfirmationResult(results,ranking,selected)
