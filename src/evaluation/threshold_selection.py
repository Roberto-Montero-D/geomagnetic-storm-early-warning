from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.dataset.temporal_splits import PERIOD_FINAL_TEST
from src.evaluation.operational import evaluate_operational_series

DEFAULT_THRESHOLD_GRID = tuple(round(x, 2) for x in np.arange(0.01, 1.00, 0.01))
DEFAULT_MAX_FAR_PER_DAY = 0.2

@dataclass(frozen=True)
class ThresholdSelectionResult:
    selected_threshold: float | None
    table: pd.DataFrame

def _validate_threshold_grid(thresholds):
    if not thresholds:
        raise ValueError("thresholds must not be empty.")
    values = tuple(float(v) for v in thresholds)
    if any(not np.isfinite(v) for v in values):
        raise ValueError("thresholds must be finite.")
    if any(v < 0 or v > 1 for v in values):
        raise ValueError("thresholds must lie in [0, 1].")
    if len(values) != len(set(values)):
        raise ValueError("thresholds must be unique.")
    if tuple(sorted(values)) != values:
        raise ValueError("thresholds must be sorted ascending.")
    return values

def _validate_development_scope(probabilities, splits):
    if not probabilities.index.isin(splits.index).all():
        raise ValueError("all probability timestamps must exist in splits.")
    if (splits.loc[probabilities.index, "period"] == PERIOD_FINAL_TEST).any():
        raise ValueError("protected Final Test predictions cannot be used for threshold selection.")

def _validate_events_do_not_include_final_test(events):
    if "start_time" not in events.columns:
        raise ValueError("events must contain start_time.")
    starts = pd.to_datetime(events["start_time"], errors="raise")
    mask = (starts >= pd.Timestamp("2022-01-01")) & (starts < pd.Timestamp("2026-01-01"))
    if mask.any():
        raise ValueError("protected Final Test events cannot be used for threshold selection.")

def select_operational_threshold(
    probabilities: pd.Series,
    events: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day: float=DEFAULT_MAX_FAR_PER_DAY,
    cooldown_hours: int=3,
    horizon_hours: int=6,
) -> ThresholdSelectionResult:
    thresholds = _validate_threshold_grid(thresholds)
    max_far = float(max_far_per_day)
    if not np.isfinite(max_far) or max_far < 0:
        raise ValueError("max_far_per_day must be finite and non-negative.")

    _validate_development_scope(probabilities, splits)
    _validate_events_do_not_include_final_test(events)

    rows = []
    for threshold in thresholds:
        _, metrics = evaluate_operational_series(
            probabilities,
            events,
            threshold=threshold,
            cooldown_hours=cooldown_hours,
            horizon_hours=horizon_hours,
        )
        far = metrics.false_alarm_rate_per_day
        feasible = (not pd.isna(far)) and far <= max_far
        rows.append({
            "threshold": threshold,
            "event_recall": metrics.event_recall,
            "false_alarm_rate_per_day": far,
            "median_lead_time": metrics.median_lead_time,
            "n_alert_episodes": metrics.n_alert_episodes,
            "n_false_alarm_episodes": metrics.n_false_alarm_episodes,
            "n_early_detections": metrics.n_early_detections,
            "n_late_detections": metrics.n_late_detections,
            "far_feasible": bool(feasible),
        })

    table = pd.DataFrame(rows)
    feasible = table.loc[table["far_feasible"], "threshold"]
    selected = None if feasible.empty else float(feasible.iloc[0])
    return ThresholdSelectionResult(selected, table)
