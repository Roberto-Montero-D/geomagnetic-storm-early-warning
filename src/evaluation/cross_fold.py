"""Cross-fold Phase 2 operational evaluation.

Fold boundaries remain explicit. Alert episodes are never constructed across
the multi-year gaps between validation windows.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.baselines.framework import DevelopmentFold
from src.evaluation.development_predictions import BASELINE_NAMES, generate_fold_baseline_predictions
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import DEFAULT_MAX_FAR_PER_DAY, DEFAULT_THRESHOLD_GRID

DETERMINISTIC_BASELINES=("B0_persistence","B1_physical")
PROBABILISTIC_BASELINES=("B2_logistic","B3_extratrees")
DETERMINISTIC_THRESHOLD=0.5

@dataclass(frozen=True)
class CrossFoldEvaluation:
    selected_thresholds: dict[str,float|None]
    fold_metrics: pd.DataFrame
    threshold_tables: dict[str,pd.DataFrame]

def _events_for_validation(events: pd.DataFrame, index: pd.DatetimeIndex, horizon_hours: int) -> pd.DataFrame:
    if len(index)==0:
        return events.iloc[0:0].copy()
    start=index.min()
    end=index.max()
    starts=pd.to_datetime(events["start_time"],errors="raise")
    # Include events whose early-warning window can overlap this validation
    # period, while excluding events that begin after it.
    mask = (
        (starts >= start)
        & (starts <= end)
    )
    return events.loc[mask].copy()

def _aggregate_far(rows: list[dict]) -> float:
    exposure=sum(r["valid_exposure_hours"] for r in rows)
    false_alarms=sum(r["n_false_alarm_episodes"] for r in rows)
    return np.nan if exposure==0 else false_alarms/(exposure/24.0)

def evaluate_development_folds(
    dataset: pd.DataFrame,
    folds: tuple[DevelopmentFold,...] | list[DevelopmentFold],
    events: pd.DataFrame,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day: float=DEFAULT_MAX_FAR_PER_DAY,
    cooldown_hours: int=3,
    horizon_hours: int=6,
) -> CrossFoldEvaluation:
    """Generate and evaluate B0-B3 across protected development folds."""
    predictions=[generate_fold_baseline_predictions(dataset,fold) for fold in folds]
    selected={name:DETERMINISTIC_THRESHOLD for name in DETERMINISTIC_BASELINES}
    threshold_tables={}

    for baseline in PROBABILISTIC_BASELINES:
        threshold_rows=[]
        for tau in thresholds:
            fold_rows=[]
            for fold,pred in zip(folds,predictions):
                p=pred.probabilities[baseline]
                ev=_events_for_validation(events,p.index,horizon_hours)
                _,m=evaluate_operational_series(p,ev,threshold=float(tau),
                    cooldown_hours=cooldown_hours,horizon_hours=horizon_hours)
                fold_rows.append({
                    "valid_exposure_hours":int(p.notna().sum()),
                    "n_false_alarm_episodes":m.n_false_alarm_episodes,
                    "event_recall":m.event_recall,
                })
            far=_aggregate_far(fold_rows)
            threshold_rows.append({
                "threshold":float(tau),
                "false_alarm_rate_per_day":far,
                "far_feasible":bool(not pd.isna(far) and far <= max_far_per_day),
            })
        table=pd.DataFrame(threshold_rows)
        threshold_tables[baseline]=table
        feasible=table.loc[table["far_feasible"],"threshold"]
        selected[baseline]=None if feasible.empty else float(feasible.iloc[0])

    metric_rows=[]
    for fold,pred in zip(folds,predictions):
        for baseline in BASELINE_NAMES:
            tau=selected[baseline]
            if tau is None:
                metric_rows.append({
                    "fold":fold.name,"baseline":baseline,"threshold":np.nan,
                    "event_recall":np.nan,"false_alarm_rate_per_day":np.nan,
                    "median_lead_time":pd.NaT,"n_alert_episodes":0,
                    "n_false_alarm_episodes":0,"n_early_detections":0,
                    "n_late_detections":0,
                    "valid_exposure_hours":int(pred.probabilities[baseline].notna().sum()),
                })
                continue
            p=pred.probabilities[baseline]
            ev=_events_for_validation(events,p.index,horizon_hours)
            _,m=evaluate_operational_series(p,ev,threshold=tau,
                cooldown_hours=cooldown_hours,horizon_hours=horizon_hours)
            metric_rows.append({
                "fold":fold.name,"baseline":baseline,"threshold":tau,
                "event_recall":m.event_recall,
                "false_alarm_rate_per_day":m.false_alarm_rate_per_day,
                "median_lead_time":m.median_lead_time,
                "n_alert_episodes":m.n_alert_episodes,
                "n_false_alarm_episodes":m.n_false_alarm_episodes,
                "n_early_detections":m.n_early_detections,
                "n_late_detections":m.n_late_detections,
                "valid_exposure_hours":int(p.notna().sum()),
            })

    return CrossFoldEvaluation(selected,pd.DataFrame(metric_rows),threshold_tables)
