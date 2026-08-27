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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.evaluation.operational import evaluate_operational_series
from src.final_test.contract import (
    PHASE8_ALERT_COOLDOWN_HOURS,
    PHASE8_FINAL_TEST_END_EXCLUSIVE,
    PHASE8_FINAL_TEST_START,
    PHASE8_HORIZON_HOURS,
    PHASE8_OPERATIONAL_THRESHOLD,
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


def _scope_phase8_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return events whose canonical start lies inside the protected period."""

    required = {
        "event_id",
        "start_time",
        "end_time",
        "boundary_status",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            "events is missing required canonical columns: "
            f"{sorted(missing)}"
        )

    starts = pd.to_datetime(
        events["start_time"],
        errors="raise",
    )

    return events.loc[
        (starts >= PHASE8_FINAL_TEST_START)
        & (starts < PHASE8_FINAL_TEST_END_EXCLUSIVE)
    ].copy()


def score_phase8_final_test(
    probabilities: pd.Series,
    targets: pd.Series,
    events: pd.DataFrame,
):
    """Score the protected test once using only frozen operational choices."""

    if not probabilities.index.equals(targets.index):
        raise ValueError(
            "probabilities and targets must have identical indices."
        )

    if probabilities.empty:
        raise ValueError("Final Test probability series is empty.")

    if probabilities.index.min() < PHASE8_FINAL_TEST_START:
        raise ValueError(
            "Final Test probabilities include pre-protected timestamps."
        )

    if probabilities.index.max() >= PHASE8_FINAL_TEST_END_EXCLUSIVE:
        raise ValueError(
            "Final Test probabilities extend beyond the protected interval."
        )

    known = targets.notna()
    if not known.any():
        raise ValueError(
            "No known Final Test targets are available for scoring."
        )

    p = probabilities.loc[known].astype(float)
    y = targets.loc[known].astype(int)

    if not y.isin([0, 1]).all():
        raise ValueError("Known Final Test targets must be binary.")

    scoped_events = _scope_phase8_events(events)

    episodes, operational = evaluate_operational_series(
        probabilities,
        scoped_events,
        threshold=PHASE8_OPERATIONAL_THRESHOLD,
        cooldown_hours=PHASE8_ALERT_COOLDOWN_HOURS,
        horizon_hours=PHASE8_HORIZON_HOURS,
    )

    detected_ids = set(
        episodes.loc[
            episodes["classification"].isin(
                ["early_detection", "late_detection"]
            )
            & episodes["associated_event_id"].notna(),
            "associated_event_id",
        ].tolist()
    )
    valid_event_ids = set(
        scoped_events["event_id"].tolist()
    )
    n_detected_events = len(
        detected_ids & valid_event_ids
    )

    lead = operational.median_lead_time
    median_lead_time_hours = (
        None
        if pd.isna(lead)
        else float(
            lead / pd.Timedelta(hours=1)
        )
    )

    pr_auc = (
        float(average_precision_score(y, p))
        if y.nunique() > 1
        else np.nan
    )
    roc_auc = (
        float(roc_auc_score(y, p))
        if y.nunique() > 1
        else np.nan
    )
    brier = float(
        brier_score_loss(y, p)
    )

    metrics = Phase8FinalMetrics(
        event_recall=float(
            operational.event_recall
        ),
        far_per_day=float(
            operational.false_alarm_rate_per_day
        ),
        median_lead_time_hours=median_lead_time_hours,
        n_events=len(scoped_events),
        n_detected_events=n_detected_events,
        n_alert_episodes=operational.n_alert_episodes,
        n_false_alarm_episodes=(
            operational.n_false_alarm_episodes
        ),
        n_early_detections=(
            operational.n_early_detections
        ),
        n_late_detections=(
            operational.n_late_detections
        ),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier_score=brier,
    )

    return episodes, metrics


def write_phase8_final_artifacts(
    output_dir: Path,
    probabilities: pd.Series,
    targets: pd.Series,
    episodes: pd.DataFrame,
    metrics: Phase8FinalMetrics,
) -> None:
    """Write immutable result artifacts after the one-time scoring call."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "probability": probabilities,
            "target": targets,
        }
    ).to_csv(
        output_dir / "final_test_predictions.csv",
        index_label="prediction_time",
    )

    episodes.to_csv(
        output_dir / "final_test_alert_episodes.csv",
        index=False,
    )

    with (
        output_dir / "final_test_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            asdict(metrics),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
