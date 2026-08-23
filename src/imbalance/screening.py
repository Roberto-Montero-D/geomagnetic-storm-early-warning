"""Phase 4 initial imbalance screening evaluator."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.baselines.extratrees import make_extratrees_model
from src.baselines.framework import DevelopmentFold, get_development_xy
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import DEFAULT_THRESHOLD_GRID, _validate_threshold_grid
from src.feature_screening.screening import _events_for_validation
from .contract import (
    PHASE4_ADVANCE_COUNT, PHASE4_EXPERIMENTS, PHASE4_EXPERIMENT_NAMES,
    PHASE4_EXTRATREES_PARAMS, PHASE4_FEATURES, PHASE4_MAX_FAR_PER_DAY,
)
from .strategies import prepare_training_data

@dataclass(frozen=True)
class ImbalanceScreeningResult:
    experiment: str
    threshold: float | None
    event_recall: float
    false_alarm_rate_per_day: float
    pr_auc: float
    operationally_feasible: bool
    validation_probability: pd.Series
    threshold_table: pd.DataFrame

@dataclass(frozen=True)
class Phase4ScreeningResult:
    experiments: dict[str, ImbalanceScreeningResult]
    ranking: pd.DataFrame
    advancing_experiments: tuple[str, ...]

def _experiment(name):
    try:
        return next(x for x in PHASE4_EXPERIMENTS if x.name == name)
    except StopIteration as exc:
        raise ValueError(f"unknown Phase 4 experiment: {name}") from exc

def _fit_experiment(dataset: pd.DataFrame, fold: DevelopmentFold, name: str):
    x_train,y_train,x_val,y_val=get_development_xy(dataset,fold,PHASE4_FEATURES)
    if x_train.isna().any().any() or x_val.isna().any().any():
        raise AssertionError("Phase 4 predictors contain missing values.")
    prepared=prepare_training_data(x_train,y_train.astype(int),_experiment(name))
    params=dict(PHASE4_EXTRATREES_PARAMS)
    model=make_extratrees_model(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    # Only class-weight experiments alter model weighting.
    if prepared.class_weight is not None:
        model.set_params(class_weight=prepared.class_weight)
    model.fit(prepared.x,prepared.y.astype(int))
    probability=pd.Series(
        model.predict_proba(x_val)[:,1],index=x_val.index,
        name="probability",dtype=float,
    )
    return probability,y_val.astype(int)

def _threshold_curve(probability,events,thresholds,max_far_per_day,cooldown_hours,horizon_hours):
    grid=_validate_threshold_grid(thresholds)
    rows=[]
    for tau in grid:
        _,m=evaluate_operational_series(
            probability,events,threshold=tau,
            cooldown_hours=cooldown_hours,horizon_hours=horizon_hours,
        )
        far=m.false_alarm_rate_per_day
        rows.append({
            "threshold":float(tau),
            "event_recall":m.event_recall,
            "false_alarm_rate_per_day":far,
            "far_feasible":bool(not pd.isna(far) and far <= max_far_per_day),
        })
    table=pd.DataFrame(rows)
    feasible=table.loc[table["far_feasible"],"threshold"]
    selected=None if feasible.empty else float(feasible.iloc[0])
    return selected,table

def evaluate_imbalance_experiment(
    dataset: pd.DataFrame, fold: DevelopmentFold, events: pd.DataFrame,
    experiment: str, *, thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day: float=PHASE4_MAX_FAR_PER_DAY,
    cooldown_hours: int=3, horizon_hours: int=6,
) -> ImbalanceScreeningResult:
    if experiment not in PHASE4_EXPERIMENT_NAMES:
        raise ValueError(f"unknown Phase 4 experiment: {experiment}")
    probability,y_val=_fit_experiment(dataset,fold,experiment)
    scoped=_events_for_validation(events,probability.index)
    pr_auc=float(average_precision_score(y_val.to_numpy(),probability.to_numpy()))
    selected,table=_threshold_curve(
        probability,scoped,thresholds,max_far_per_day,cooldown_hours,horizon_hours
    )
    if selected is None:
        return ImbalanceScreeningResult(
            experiment,None,np.nan,np.nan,pr_auc,False,probability,table
        )
    _,m=evaluate_operational_series(
        probability,scoped,threshold=selected,
        cooldown_hours=cooldown_hours,horizon_hours=horizon_hours,
    )
    return ImbalanceScreeningResult(
        experiment,selected,m.event_recall,m.false_alarm_rate_per_day,
        pr_auc,True,probability,table
    )

def rank_imbalance_experiments(experiments):
    order={name:i for i,name in enumerate(PHASE4_EXPERIMENT_NAMES)}
    rows=[]
    for name in PHASE4_EXPERIMENT_NAMES:
        r=experiments[name]
        rows.append({
            "experiment":name,"threshold":r.threshold,
            "event_recall":r.event_recall,
            "false_alarm_rate_per_day":r.false_alarm_rate_per_day,
            "pr_auc":r.pr_auc,
            "operationally_feasible":r.operationally_feasible,
            "_order":order[name],
        })
    ranking=pd.DataFrame(rows).sort_values(
        ["operationally_feasible","event_recall","pr_auc",
         "false_alarm_rate_per_day","_order"],
        ascending=[False,False,False,True,True],
        na_position="last",kind="mergesort",
    ).reset_index(drop=True)
    feasible=ranking.loc[ranking.operationally_feasible,"experiment"]
    advancing=tuple(feasible.iloc[:min(PHASE4_ADVANCE_COUNT,len(feasible))])
    return ranking.drop(columns="_order"),advancing

def evaluate_phase4_screening(
    dataset,fold,events,*,thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day=PHASE4_MAX_FAR_PER_DAY,
):
    results={}
    for name in PHASE4_EXPERIMENT_NAMES:
        results[name]=evaluate_imbalance_experiment(
            dataset,fold,events,name,thresholds=thresholds,
            max_far_per_day=max_far_per_day,
        )
    ranking,advancing=rank_imbalance_experiments(results)
    return Phase4ScreeningResult(results,ranking,advancing)
