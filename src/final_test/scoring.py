"""Single-use Phase 8 protected Final Test scoring primitives.

No model selection or threshold optimization exists here. The caller supplies
the already-frozen probability series and the frozen canonical event truth.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.operational import evaluate_operational_series
from src.final_test.contract import (
    PHASE8_HORIZON_HOURS, PHASE8_OPERATIONAL_THRESHOLD,
)

@dataclass(frozen=True)
class Phase8FinalMetrics:
    event_recall: float
    far_per_day: float
    median_lead_time_hours: float | None
    n_events: int
    n_detected_events: int
    n_alert_episodes: int
    n_false_alarm_episodes: int
    n_early_detections: int
    n_late_detections: int
    pr_auc: float
    roc_auc: float
    brier_score: float

def score_phase8_final_test(probabilities: pd.Series, targets: pd.Series,
                            events: pd.DataFrame):
    if not probabilities.index.equals(targets.index):
        raise ValueError("probabilities and targets must have identical indices.")
    known=targets.notna()
    if not known.any():
        raise ValueError("No known Final Test targets are available for scoring.")
    p=probabilities.loc[known].astype(float)
    y=targets.loc[known].astype(int)
    scoped_events=events[
        (events["storm_start"] >= probabilities.index.min()) &
        (events["storm_start"] <= probabilities.index.max())
    ].copy()
    episodes, op=evaluate_operational_series(
        probabilities, scoped_events,
        threshold=PHASE8_OPERATIONAL_THRESHOLD,
        cooldown_hours=3,
        horizon_hours=PHASE8_HORIZON_HOURS,
    )
    detected=set(episodes.loc[
        episodes["classification"].isin(["early_detection","late_detection"])
        & episodes["associated_storm_id"].notna(),
        "associated_storm_id"
    ].tolist())
    leads=op.median_lead_time
    lead_hours=None if pd.isna(leads) else float(leads / pd.Timedelta(hours=1))
    pr=float(average_precision_score(y,p)) if y.nunique()>1 else np.nan
    roc=float(roc_auc_score(y,p)) if y.nunique()>1 else np.nan
    brier=float(brier_score_loss(y,p))
    metrics=Phase8FinalMetrics(
        event_recall=float(op.event_recall),
        far_per_day=float(op.false_alarm_rate_per_day),
        median_lead_time_hours=lead_hours,
        n_events=len(scoped_events),
        n_detected_events=len(detected),
        n_alert_episodes=op.n_alert_episodes,
        n_false_alarm_episodes=op.n_false_alarm_episodes,
        n_early_detections=op.n_early_detections,
        n_late_detections=op.n_late_detections,
        pr_auc=pr, roc_auc=roc, brier_score=brier,
    )
    return episodes, metrics

def write_phase8_final_artifacts(output_dir: Path, probabilities: pd.Series,
                                 targets: pd.Series, episodes: pd.DataFrame,
                                 metrics: Phase8FinalMetrics) -> None:
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"probability":probabilities,"target":targets}).to_csv(
        output_dir/"final_test_predictions.csv", index_label="prediction_time")
    episodes.to_csv(output_dir/"final_test_alert_episodes.csv",index=False)
    with (output_dir/"final_test_metrics.json").open("w",encoding="utf-8") as f:
        json.dump(asdict(metrics),f,indent=2,sort_keys=True); f.write("\n")
