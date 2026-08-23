"""Immutable Phase 3 feature-selection outcome.

This module records the result of the frozen Phase 3 screening and
walk-forward confirmation protocol. Later phases must import this contract
rather than re-selecting a Phase 3 feature set from observed results.
"""

from __future__ import annotations

from src.feature_screening.manifests import PHASE3_FEATURE_SETS

PHASE3_SELECTED_EXPERIMENT = "A"
PHASE3_SELECTED_FEATURES = tuple(PHASE3_FEATURE_SETS[PHASE3_SELECTED_EXPERIMENT])

# Development-only evidence from the official Phase 3 confirmation run.
# These values document the frozen decision; they are not tuning inputs.
PHASE3_CONFIRMATION_EVIDENCE = {
    "walk_forward_1": {
        "threshold": 0.05,
        "event_recall": 0.5333333333333333,
        "false_alarm_rate_per_day": 0.1789,
        "pr_auc": 0.2470,
    },
    "walk_forward_2": {
        "threshold": 0.08,
        "event_recall": 0.75,
        "false_alarm_rate_per_day": 0.1993,
        "pr_auc": 0.2702,
    },
}

PHASE3_SELECTION_STATUS = "frozen"
