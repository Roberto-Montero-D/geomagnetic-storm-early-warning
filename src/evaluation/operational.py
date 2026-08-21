"""Phase 2 operational evaluation adapters.

Reuses the canonical Phase 0 alert/event definitions. This module only
normalizes deterministic baseline predictions and probabilistic model outputs
into the probability-series interface required by src.definitions.alerts.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.definitions.alerts import (
    identify_alerts, associate_alerts_with_events, event_recall,
    false_alarm_rate_per_day, early_detection_lead_times,
)

@dataclass(frozen=True)
class OperationalMetrics:
    event_recall: float
    false_alarm_rate_per_day: float
    median_lead_time: pd.Timedelta | pd.NaT
    n_alert_episodes: int
    n_false_alarm_episodes: int
    n_early_detections: int
    n_late_detections: int

def binary_predictions_as_probabilities(predictions: pd.Series) -> pd.Series:
    """Map deterministic 0/1/NA baseline output to 0.0/1.0/NaN."""
    if not isinstance(predictions,pd.Series):
        raise TypeError("predictions must be a pandas Series.")
    numeric=pd.to_numeric(predictions,errors="raise")
    finite=numeric.notna()
    if not numeric.loc[finite].isin([0,1]).all():
        raise ValueError("deterministic predictions must contain only 0, 1, or missing values.")
    return pd.Series(numeric.astype(float),index=predictions.index,
                     name="probability",dtype=float)

def evaluate_operational_series(probabilities: pd.Series, events: pd.DataFrame, *,
                                threshold: float,
                                cooldown_hours: int=3,
                                horizon_hours: int=6) -> tuple[pd.DataFrame, OperationalMetrics]:
    """Evaluate one already-scoped development prediction series."""
    episodes=identify_alerts(probabilities,threshold,cooldown_hours=cooldown_hours)
    associated=associate_alerts_with_events(episodes,events,horizon_hours=horizon_hours)
    recall=event_recall(associated,events)
    far=false_alarm_rate_per_day(associated,probabilities)
    leads=early_detection_lead_times(associated)
    median=pd.NaT if leads.empty else leads.median()
    metrics=OperationalMetrics(
        event_recall=float(recall) if not pd.isna(recall) else np.nan,
        false_alarm_rate_per_day=float(far) if not pd.isna(far) else np.nan,
        median_lead_time=median,
        n_alert_episodes=len(associated),
        n_false_alarm_episodes=int((associated["classification"]=="false_alarm").sum()),
        n_early_detections=int((associated["classification"]=="early_detection").sum()),
        n_late_detections=int((associated["classification"]=="late_detection").sum()),
    )
    return associated,metrics
