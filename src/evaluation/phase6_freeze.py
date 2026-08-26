"""Immutable Phase 6 operational handoff.

This module records the already-selected development-only Phase 6 operating
point. Phase 8 and later code must import this handoff rather than re-running
threshold selection or retyping the selected model/threshold independently.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.oof_predictions import PHASE6_SELECTED_CONFIG_ID


PHASE6_SELECTED_THRESHOLD = 0.10
PHASE6_OOF_ROWS = 25_873
PHASE6_EVENT_RECALL = 21 / 31
PHASE6_N_EVENTS = 31
PHASE6_N_DETECTED_EVENTS = 21
PHASE6_FALSE_ALARM_RATE_PER_DAY = 0.18552158621033587
PHASE6_FOLD_SELECTED_THRESHOLDS = {
    "walk_forward_1": 0.07,
    "walk_forward_2": 0.16,
}
PHASE6_STABILITY_THRESHOLDS = (0.10, 0.11, 0.12, 0.13)
PHASE6_PROTECTED_FINAL_TEST_SCORED = False


@dataclass(frozen=True)
class FrozenPhase6Decision:
    config_id: str
    threshold: float
    oof_rows: int
    event_recall: float
    n_events: int
    n_detected_events: int
    false_alarm_rate_per_day: float
    fold_selected_thresholds: tuple[tuple[str, float], ...]
    stability_thresholds: tuple[float, ...]
    protected_final_test_scored: bool


PHASE6_FROZEN_DECISION = FrozenPhase6Decision(
    config_id=PHASE6_SELECTED_CONFIG_ID,
    threshold=PHASE6_SELECTED_THRESHOLD,
    oof_rows=PHASE6_OOF_ROWS,
    event_recall=PHASE6_EVENT_RECALL,
    n_events=PHASE6_N_EVENTS,
    n_detected_events=PHASE6_N_DETECTED_EVENTS,
    false_alarm_rate_per_day=PHASE6_FALSE_ALARM_RATE_PER_DAY,
    fold_selected_thresholds=tuple(PHASE6_FOLD_SELECTED_THRESHOLDS.items()),
    stability_thresholds=PHASE6_STABILITY_THRESHOLDS,
    protected_final_test_scored=PHASE6_PROTECTED_FINAL_TEST_SCORED,
)


def validate_phase6_freeze() -> None:
    """Assert the immutable Phase 6 handoff."""

    assert PHASE6_FROZEN_DECISION.config_id == "lightgbm_lr0.1_leaves127"
    assert PHASE6_FROZEN_DECISION.threshold == 0.10
    assert PHASE6_FROZEN_DECISION.oof_rows == 25_873
    assert PHASE6_FROZEN_DECISION.n_events == 31
    assert PHASE6_FROZEN_DECISION.n_detected_events == 21
    assert PHASE6_FROZEN_DECISION.event_recall == 21 / 31
    assert PHASE6_FROZEN_DECISION.false_alarm_rate_per_day <= 0.2
    assert dict(PHASE6_FROZEN_DECISION.fold_selected_thresholds) == {
        "walk_forward_1": 0.07,
        "walk_forward_2": 0.16,
    }
    assert PHASE6_FROZEN_DECISION.stability_thresholds == (
        0.10,
        0.11,
        0.12,
        0.13,
    )
    assert PHASE6_FROZEN_DECISION.protected_final_test_scored is False


validate_phase6_freeze()
