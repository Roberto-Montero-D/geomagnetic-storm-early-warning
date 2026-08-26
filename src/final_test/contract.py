"""Immutable Phase 8 protected Final Test contract.

This module contains no model fitting, prediction generation, event scoring,
metric calculation, or result inspection. Its only job is to bind the frozen
Phase 0-7 handoffs into one machine-readable contract before the protected
2022-2025 Final Test is opened.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    assign_temporal_periods,
)
from src.evaluation.phase6_freeze import PHASE6_FROZEN_DECISION
from src.feature_screening.freeze import PHASE3_SELECTED_FEATURES
from src.imbalance.freeze import PHASE4_FROZEN_DECISION


PHASE8_STORM_THRESHOLD = 5.0
PHASE8_HORIZON_HOURS = 6
PHASE8_EVENT_TERMINATION_HOURS = 6
PHASE8_ALERT_COOLDOWN_HOURS = 3
PHASE8_MAX_FAR_PER_DAY = 0.2

PHASE8_FEATURES = tuple(PHASE3_SELECTED_FEATURES)
PHASE8_IMBALANCE_EXPERIMENT = PHASE4_FROZEN_DECISION.experiment
PHASE8_USE_RESAMPLING = PHASE4_FROZEN_DECISION.use_resampling
PHASE8_CLASS_WEIGHT = PHASE4_FROZEN_DECISION.class_weight

PHASE8_MODEL_CONFIG_ID = PHASE6_FROZEN_DECISION.config_id
PHASE8_OPERATIONAL_THRESHOLD = PHASE6_FROZEN_DECISION.threshold

PHASE8_TRAIN_START = pd.Timestamp("1996-01-01 00:00")
PHASE8_TRAIN_END_EXCLUSIVE = pd.Timestamp("2022-01-01 00:00")
PHASE8_FINAL_TEST_START = pd.Timestamp("2022-01-01 00:00")
PHASE8_FINAL_TEST_END_EXCLUSIVE = pd.Timestamp("2026-01-01 00:00")

PHASE8_PRIMARY_EXPERIMENT_ID = "t5_h6"
PHASE8_SINGLE_USE = True
PHASE8_RESULTS_MAY_TRIGGER_RETUNING = False


@dataclass(frozen=True)
class FrozenPhase8Contract:
    experiment_id: str
    storm_threshold: float
    horizon_hours: int
    event_termination_hours: int
    alert_cooldown_hours: int
    max_far_per_day: float
    features: tuple[str, ...]
    imbalance_experiment: str
    use_resampling: bool
    class_weight: int | float | None
    model_config_id: str
    operational_threshold: float
    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    final_test_start: pd.Timestamp
    final_test_end_exclusive: pd.Timestamp
    single_use: bool
    results_may_trigger_retuning: bool


PHASE8_FROZEN_CONTRACT = FrozenPhase8Contract(
    experiment_id=PHASE8_PRIMARY_EXPERIMENT_ID,
    storm_threshold=PHASE8_STORM_THRESHOLD,
    horizon_hours=PHASE8_HORIZON_HOURS,
    event_termination_hours=PHASE8_EVENT_TERMINATION_HOURS,
    alert_cooldown_hours=PHASE8_ALERT_COOLDOWN_HOURS,
    max_far_per_day=PHASE8_MAX_FAR_PER_DAY,
    features=PHASE8_FEATURES,
    imbalance_experiment=PHASE8_IMBALANCE_EXPERIMENT,
    use_resampling=PHASE8_USE_RESAMPLING,
    class_weight=PHASE8_CLASS_WEIGHT,
    model_config_id=PHASE8_MODEL_CONFIG_ID,
    operational_threshold=PHASE8_OPERATIONAL_THRESHOLD,
    train_start=PHASE8_TRAIN_START,
    train_end_exclusive=PHASE8_TRAIN_END_EXCLUSIVE,
    final_test_start=PHASE8_FINAL_TEST_START,
    final_test_end_exclusive=PHASE8_FINAL_TEST_END_EXCLUSIVE,
    single_use=PHASE8_SINGLE_USE,
    results_may_trigger_retuning=PHASE8_RESULTS_MAY_TRIGGER_RETUNING,
)


def validate_phase8_contract() -> None:
    """Assert that the Final Test handoff matches every frozen Phase 0-7 decision."""

    c = PHASE8_FROZEN_CONTRACT

    assert c.experiment_id == "t5_h6"
    assert c.storm_threshold == 5.0
    assert c.horizon_hours == 6
    assert c.event_termination_hours == 6
    assert c.alert_cooldown_hours == 3
    assert c.max_far_per_day == 0.2

    assert c.features == tuple(PHASE3_SELECTED_FEATURES)
    assert len(c.features) == 10

    assert c.imbalance_experiment == PHASE4_FROZEN_DECISION.experiment == "none"
    assert c.use_resampling is False
    assert c.class_weight is None

    assert c.model_config_id == PHASE6_FROZEN_DECISION.config_id
    assert c.model_config_id == "lightgbm_lr0.1_leaves127"
    assert c.operational_threshold == PHASE6_FROZEN_DECISION.threshold == 0.10

    assert c.train_start == pd.Timestamp("1996-01-01 00:00")
    assert c.train_end_exclusive == c.final_test_start
    assert c.final_test_start == pd.Timestamp("2022-01-01 00:00")
    assert c.final_test_end_exclusive == pd.Timestamp("2026-01-01 00:00")
    assert c.single_use is True
    assert c.results_may_trigger_retuning is False

    # Verify the public Phase 1 temporal API recognizes the protected bounds.
    boundary_times = pd.DatetimeIndex(
        [
            c.final_test_start,
            c.final_test_end_exclusive - pd.Timedelta(hours=1),
        ]
    )
    assigned = assign_temporal_periods(boundary_times)
    assert assigned["period"].eq(PERIOD_FINAL_TEST).all()
    assert assigned["is_final_test"].all()


validate_phase8_contract()
