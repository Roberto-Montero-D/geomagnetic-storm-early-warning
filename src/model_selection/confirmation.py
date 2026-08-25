"""Phase 5 walk-forward confirmation for the three frozen family winners."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.baselines.framework import DevelopmentFold, get_development_xy
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import DEFAULT_THRESHOLD_GRID
from src.feature_screening.screening import _events_for_validation
from src.imbalance.screening import _threshold_curve

from .contract import PHASE5_FEATURES, PHASE5_MAX_FAR_PER_DAY
from .factories import make_phase5_model_by_id

PHASE5_CONFIRMATION_CANDIDATES=(
    "extratrees_n100_dnone",
    "lightgbm_lr0.1_leaves127",
    "xgboost_lr0.1_d9",
)
PHASE5_CONFIRMATION_FOLDS=("walk_forward_1","walk_forward_2")


@dataclass(frozen=True)
class ConfirmationFoldResult:
    fold_name: str
    config_id: str
    threshold: float | None
    event_recall: float
    false_alarm_rate_per_day: float
    pr_auc: float
    operationally_feasible: bool
    threshold_table: pd.DataFrame


@dataclass(frozen=True)
class Phase5ConfirmationResult:
    fold_results: dict[tuple[str,str],ConfirmationFoldResult]
    ranking: pd.DataFrame
    selected_config_id: str | None


def _validate_confirmation_fold(fold: DevelopmentFold,splits: pd.DataFrame,fold_name: str):
    if fold.name != fold_name:
        raise ValueError(f"{fold_name} confirmation received fold {fold.name!r}.")
    if fold_name not in PHASE5_CONFIRMATION_FOLDS:
        raise ValueError(f"Unsupported Phase 5 confirmation fold: {fold_name}")

    expected={
        "walk_forward_1":({"initial_train","validation_1"},{"validation_2"}),
        "walk_forward_2":(
            {"initial_train","validation_1","validation_2"},
            {"validation_3"},
        ),
    }
    train_expected,val_expected=expected[fold_name]
    train_periods=set(splits.loc[fold.train_index,"period"].astype(str).unique())
    val_periods=set(splits.loc[fold.validation_index,"period"].astype(str).unique())

    if "final_test" in train_periods or "final_test" in val_periods:
        raise ValueError("Phase 5 confirmation must never access protected Final Test rows.")
    if train_periods != train_expected:
        raise ValueError(f"{fold_name} training rows do not match the frozen temporal contract.")
    if val_periods != val_expected:
        raise ValueError(f"{fold_name} validation rows do not match the frozen temporal contract.")
    if fold.train_index.intersection(fold.validation_index).size:
        raise ValueError(f"{fold_name} train and validation indices overlap.")
    if len(fold.train_index) and len(fold.validation_index):
        if fold.train_index.max() >= fold.validation_index.min():
            raise ValueError(f"{fold_name} is not strictly chronological.")


def evaluate_confirmation_fold(dataset,fold,events,splits,fold_name,config_id,*,
                               thresholds=DEFAULT_THRESHOLD_GRID,
                               max_far_per_day=PHASE5_MAX_FAR_PER_DAY):
    if config_id not in PHASE5_CONFIRMATION_CANDIDATES:
        raise ValueError(f"Configuration {config_id} did not advance from Phase 5 screening.")
    _validate_confirmation_fold(fold,splits,fold_name)
    xtr,ytr,xv,yv=get_development_xy(dataset,fold,PHASE5_FEATURES)
    if xtr.isna().any().any() or xv.isna().any().any():
        raise AssertionError("Phase 5 confirmation predictors contain missing values.")
    model=make_phase5_model_by_id(config_id)
    model.fit(xtr,ytr.astype(int))
    prob=pd.Series(model.predict_proba(xv)[:,1],index=xv.index,name="probability",dtype=float)
    scoped=_events_for_validation(events,prob.index)
    pr=float(average_precision_score(yv.astype(int),prob))
    tau,table=_threshold_curve(prob,scoped,thresholds,max_far_per_day,3,6)
    if tau is None:
        return ConfirmationFoldResult(fold_name,config_id,None,np.nan,np.nan,pr,False,table)
    _,m=evaluate_operational_series(prob,scoped,threshold=tau,cooldown_hours=3,horizon_hours=6)
    return ConfirmationFoldResult(fold_name,config_id,tau,m.event_recall,
                                  m.false_alarm_rate_per_day,pr,True,table)


def rank_confirmation_candidates(results):
    rows=[]
    order={c:i for i,c in enumerate(PHASE5_CONFIRMATION_CANDIDATES)}
    for cid in PHASE5_CONFIRMATION_CANDIDATES:
        rs=[results[(fold,cid)] for fold in PHASE5_CONFIRMATION_FOLDS]
        recalls=[r.event_recall for r in rs]
        fars=[r.false_alarm_rate_per_day for r in rs]
        prs=[r.pr_auc for r in rs]
        feasible=all(r.operationally_feasible for r in rs)
        rows.append({
            "config_id":cid,
            "feasible_both_folds":feasible,
            "worst_fold_event_recall":min(recalls) if feasible else np.nan,
            "mean_event_recall":float(np.mean(recalls)) if feasible else np.nan,
            "mean_pr_auc":float(np.mean(prs)),
            "mean_false_alarm_rate_per_day":float(np.mean(fars)) if feasible else np.nan,
            "_order":order[cid],
        })
    return pd.DataFrame(rows).sort_values(
        ["feasible_both_folds","worst_fold_event_recall","mean_event_recall",
         "mean_pr_auc","mean_false_alarm_rate_per_day","_order"],
        ascending=[False,False,False,False,True,True],
        na_position="last",kind="mergesort"
    ).reset_index(drop=True).drop(columns="_order")


def evaluate_phase5_confirmation(dataset,folds,events,splits):
    results={}
    for fold_name in PHASE5_CONFIRMATION_FOLDS:
        for cid in PHASE5_CONFIRMATION_CANDIDATES:
            results[(fold_name,cid)]=evaluate_confirmation_fold(
                dataset,folds[fold_name],events,splits,fold_name,cid)
    ranking=rank_confirmation_candidates(results)
    feasible=ranking[ranking.feasible_both_folds]
    selected=None if feasible.empty else str(feasible.iloc[0].config_id)
    return Phase5ConfirmationResult(results,ranking,selected)
