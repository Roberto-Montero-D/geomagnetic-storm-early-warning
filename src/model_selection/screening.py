"""Phase 5 initial model-selection screening evaluator."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.baselines.framework import DevelopmentFold, get_development_xy
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import DEFAULT_THRESHOLD_GRID
from src.feature_screening.screening import _events_for_validation
from src.imbalance.screening import _threshold_curve
from .contract import (
    PHASE5_ADVANCE_PER_FAMILY, PHASE5_CONFIGURATIONS, PHASE5_CONFIG_IDS,
    PHASE5_FEATURES, PHASE5_MAX_FAR_PER_DAY,
)
from .factories import configuration_by_id, make_phase5_model

@dataclass(frozen=True)
class ModelScreeningResult:
    config_id: str
    family: str
    threshold: float | None
    event_recall: float
    false_alarm_rate_per_day: float
    pr_auc: float
    operationally_feasible: bool
    validation_probability: pd.Series
    threshold_table: pd.DataFrame

@dataclass(frozen=True)
class Phase5ScreeningResult:
    configurations: dict[str, ModelScreeningResult]
    family_rankings: dict[str, pd.DataFrame]
    advancing_configurations: tuple[str, ...]

def _fit_configuration(dataset: pd.DataFrame, fold: DevelopmentFold, config_id: str):
    config=configuration_by_id(config_id)
    xtr,ytr,xv,yv=get_development_xy(dataset,fold,PHASE5_FEATURES)
    if xtr.isna().any().any() or xv.isna().any().any():
        raise AssertionError("Phase 5 predictors contain missing values.")
    if len(np.unique(ytr.astype(int).to_numpy())) != 2:
        raise ValueError("Phase 5 training target must contain both classes.")
    model=make_phase5_model(config)
    model.fit(xtr,ytr.astype(int))
    prob=pd.Series(model.predict_proba(xv)[:,1],index=xv.index,name="probability",dtype=float)
    if not prob.between(0,1).all():
        raise AssertionError("Phase 5 validation probabilities must lie in [0, 1].")
    return prob,yv.astype(int)

def evaluate_model_configuration(dataset,fold,events,config_id,*,thresholds=DEFAULT_THRESHOLD_GRID,
                                 max_far_per_day=PHASE5_MAX_FAR_PER_DAY,
                                 cooldown_hours=3,horizon_hours=6):
    if config_id not in PHASE5_CONFIG_IDS:
        raise ValueError(f"unknown Phase 5 configuration: {config_id}")
    config=configuration_by_id(config_id)
    prob,yv=_fit_configuration(dataset,fold,config_id)
    scoped=_events_for_validation(events,prob.index)
    pr=float(average_precision_score(yv.to_numpy(),prob.to_numpy()))
    tau,table=_threshold_curve(prob,scoped,thresholds,max_far_per_day,cooldown_hours,horizon_hours)
    if tau is None:
        return ModelScreeningResult(config_id,config.family,None,np.nan,np.nan,pr,False,prob,table)
    _,m=evaluate_operational_series(prob,scoped,threshold=tau,
        cooldown_hours=cooldown_hours,horizon_hours=horizon_hours)
    return ModelScreeningResult(config_id,config.family,tau,m.event_recall,
        m.false_alarm_rate_per_day,pr,True,prob,table)

def rank_family_configurations(configurations,family):
    frozen_order={c.config_id:i for i,c in enumerate(PHASE5_CONFIGURATIONS)}
    expected=[c.config_id for c in PHASE5_CONFIGURATIONS if c.family==family]
    missing=[cid for cid in expected if cid not in configurations]
    if missing:
        raise ValueError(f"Missing Phase 5 {family} screening results: {missing}")
    rows=[]
    for cid in expected:
        r=configurations[cid]
        if r.family != family:
            raise ValueError(f"Configuration {cid} has unexpected family {r.family!r}; expected {family!r}.")
        rows.append({"config_id":cid,"family":family,"threshold":r.threshold,
            "event_recall":r.event_recall,"false_alarm_rate_per_day":r.false_alarm_rate_per_day,
            "pr_auc":r.pr_auc,"operationally_feasible":r.operationally_feasible,
            "_order":frozen_order[cid]})
    return pd.DataFrame(rows).sort_values(
        ["operationally_feasible","event_recall","pr_auc","false_alarm_rate_per_day","_order"],
        ascending=[False,False,False,True,True],na_position="last",kind="mergesort"
    ).reset_index(drop=True).drop(columns="_order")

def advance_family_winners(configurations):
    rankings={}; advancing=[]
    for family in ("extratrees","lightgbm","xgboost"):
        ranking=rank_family_configurations(configurations,family)
        rankings[family]=ranking
        feasible=ranking.loc[ranking.operationally_feasible,"config_id"]
        if len(feasible) >= PHASE5_ADVANCE_PER_FAMILY:
            advancing.extend(feasible.iloc[:PHASE5_ADVANCE_PER_FAMILY].tolist())
    return rankings,tuple(advancing)

def evaluate_phase5_screening(dataset,fold,events,*,thresholds=DEFAULT_THRESHOLD_GRID,
                              max_far_per_day=PHASE5_MAX_FAR_PER_DAY):
    results={}
    for config in PHASE5_CONFIGURATIONS:
        results[config.config_id]=evaluate_model_configuration(
            dataset,fold,events,config.config_id,thresholds=thresholds,
            max_far_per_day=max_far_per_day)
    rankings,advancing=advance_family_winners(results)
    return Phase5ScreeningResult(results,rankings,advancing)
