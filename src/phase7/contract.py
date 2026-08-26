"""Frozen Phase 7 horizon/severity experiment registry.

Phase 7 changes retrospective truth through T and H while preserving the
already-frozen predictor, imbalance, model, temporal, and operational
contracts from Phases 3-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.definitions.events import identify_events
from src.feature_screening.freeze import PHASE3_SELECTED_FEATURES
from src.imbalance.freeze import PHASE4_FROZEN_DECISION
from src.targets.event_window import build_event_window_target


PHASE7_MODEL_CONFIG_ID = "lightgbm_lr0.1_leaves127"
PHASE7_IMBALANCE_EXPERIMENT = PHASE4_FROZEN_DECISION.experiment
PHASE7_FEATURES = tuple(PHASE3_SELECTED_FEATURES)

PHASE7_TERMINATION_HOURS = 6
PHASE7_ALERT_COOLDOWN_HOURS = 3
PHASE7_MAX_FAR_PER_DAY = 0.2

PHASE7_THRESHOLD_MIN = 0.01
PHASE7_THRESHOLD_MAX = 0.99
PHASE7_THRESHOLD_STEP = 0.01

PHASE7_PRIMARY_CONTROL_ID = "t5_h6"


@dataclass(frozen=True)
class Phase7Experiment:
    """One pre-authorized Phase 7 truth configuration."""

    experiment_id: str
    threshold: float
    horizon_hours: int
    is_primary_control: bool = False


PHASE7_EXPERIMENTS = (
    Phase7Experiment("t5_h3", 5.0, 3),
    Phase7Experiment("t5_h6", 5.0, 6, True),
    Phase7Experiment("t5_h12", 5.0, 12),
    Phase7Experiment("t5_h24", 5.0, 24),
    Phase7Experiment("t6_h6", 6.0, 6),
    Phase7Experiment("t7_h6", 7.0, 6),
)

PHASE7_EXPERIMENT_IDS = tuple(
    experiment.experiment_id
    for experiment in PHASE7_EXPERIMENTS
)

_EXPERIMENT_BY_ID = {
    experiment.experiment_id: experiment
    for experiment in PHASE7_EXPERIMENTS
}


def get_phase7_experiment(
    experiment_id: str,
) -> Phase7Experiment:
    """Return one frozen Phase 7 experiment by ID."""

    try:
        return _EXPERIMENT_BY_ID[experiment_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Phase 7 experiment: {experiment_id!r}."
        ) from exc


def build_phase7_target(
    kp_intervals: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    experiment: str | Phase7Experiment,
    *,
    return_audit: bool = False,
):
    """Build experiment-specific target truth using the canonical builder."""

    spec = (
        get_phase7_experiment(experiment)
        if isinstance(experiment, str)
        else experiment
    )
    if not isinstance(spec, Phase7Experiment):
        raise TypeError(
            "experiment must be a Phase7Experiment or registered experiment ID."
        )

    return build_event_window_target(
        kp_intervals,
        prediction_times,
        threshold=spec.threshold,
        horizon_hours=spec.horizon_hours,
        return_audit=return_audit,
    )


def build_phase7_events(
    kp_intervals: pd.DataFrame,
    experiment: str | Phase7Experiment,
) -> pd.DataFrame:
    """Build experiment-specific event truth with frozen Z=6 h."""

    spec = (
        get_phase7_experiment(experiment)
        if isinstance(experiment, str)
        else experiment
    )
    if not isinstance(spec, Phase7Experiment):
        raise TypeError(
            "experiment must be a Phase7Experiment or registered experiment ID."
        )

    return identify_events(
        kp_intervals,
        threshold=spec.threshold,
        termination_hours=PHASE7_TERMINATION_HOURS,
    )


def validate_phase7_contract() -> None:
    """Assert the immutable Phase 7.0-7.3 registry contract."""

    expected = (
        ("t5_h3", 5.0, 3),
        ("t5_h6", 5.0, 6),
        ("t5_h12", 5.0, 12),
        ("t5_h24", 5.0, 24),
        ("t6_h6", 6.0, 6),
        ("t7_h6", 7.0, 6),
    )
    actual = tuple(
        (
            experiment.experiment_id,
            experiment.threshold,
            experiment.horizon_hours,
        )
        for experiment in PHASE7_EXPERIMENTS
    )

    assert actual == expected
    assert len(PHASE7_EXPERIMENTS) == 6
    assert len(set(PHASE7_EXPERIMENT_IDS)) == 6
    assert len(
        {
            (experiment.threshold, experiment.horizon_hours)
            for experiment in PHASE7_EXPERIMENTS
        }
    ) == 6

    controls = tuple(
        experiment.experiment_id
        for experiment in PHASE7_EXPERIMENTS
        if experiment.is_primary_control
    )
    assert controls == (PHASE7_PRIMARY_CONTROL_ID,)

    assert PHASE7_FEATURES == tuple(PHASE3_SELECTED_FEATURES)
    assert len(PHASE7_FEATURES) == 10
    assert PHASE7_IMBALANCE_EXPERIMENT == "none"
    assert PHASE7_MODEL_CONFIG_ID == "lightgbm_lr0.1_leaves127"
    assert PHASE7_TERMINATION_HOURS == 6
    assert PHASE7_ALERT_COOLDOWN_HOURS == 3
    assert PHASE7_MAX_FAR_PER_DAY == 0.2
    assert PHASE7_THRESHOLD_MIN == 0.01
    assert PHASE7_THRESHOLD_MAX == 0.99
    assert PHASE7_THRESHOLD_STEP == 0.01


validate_phase7_contract()
