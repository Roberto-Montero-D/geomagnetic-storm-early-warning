"""Frozen Phase 4 imbalance decision.

This module is the canonical machine-readable handoff from Phase 4 to all
subsequent model-development phases. Later phases must import this decision
rather than re-selecting an imbalance treatment from development results.
"""
from __future__ import annotations

from dataclasses import dataclass

PHASE4_SELECTED_EXPERIMENT = "none"
PHASE4_USE_RESAMPLING = False
PHASE4_CLASS_WEIGHT = None

PHASE4_SCREENING_ADVANCERS = (
    "undersample_10_to_1",
    "none",
    "class_weight_1",
)

PHASE4_DECISION_RULE = (
    "feasible in both confirmation folds; highest minimum Event Recall; "
    "highest mean Event Recall; highest mean PR-AUC; lowest mean FAR/day; "
    "frozen candidate order"
)


@dataclass(frozen=True)
class FrozenImbalanceDecision:
    experiment: str
    use_resampling: bool
    class_weight: int | float | None
    screening_advancers: tuple[str, ...]
    decision_rule: str


PHASE4_FROZEN_DECISION = FrozenImbalanceDecision(
    experiment=PHASE4_SELECTED_EXPERIMENT,
    use_resampling=PHASE4_USE_RESAMPLING,
    class_weight=PHASE4_CLASS_WEIGHT,
    screening_advancers=PHASE4_SCREENING_ADVANCERS,
    decision_rule=PHASE4_DECISION_RULE,
)
