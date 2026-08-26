"""Phase 7 experiment-specific operational threshold recalibration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.evaluation.phase6_thresholds import (
    DEFAULT_STABILITY_MIN_FAR_PER_DAY,
    Phase6ThresholdOptimization,
    optimize_phase6_threshold,
)
from src.evaluation.threshold_selection import DEFAULT_THRESHOLD_GRID
from src.phase7.contract import (
    PHASE7_ALERT_COOLDOWN_HOURS,
    PHASE7_EXPERIMENT_IDS,
    PHASE7_MAX_FAR_PER_DAY,
    Phase7Experiment,
    build_phase7_events,
    get_phase7_experiment,
)
from src.phase7.oof import Phase7OOFPredictions


@dataclass(frozen=True)
class Phase7ThresholdOptimization:
    """Threshold-selection result for one frozen Phase 7 experiment."""

    experiment_id: str
    selected_threshold: float | None
    global_threshold_table: pd.DataFrame
    fold_threshold_table: pd.DataFrame
    fold_selected_thresholds: dict[str, float | None]
    stability_thresholds: tuple[float, ...]


def _resolve_experiment(
    experiment: str | Phase7Experiment,
) -> Phase7Experiment:
    if isinstance(experiment, str):
        return get_phase7_experiment(experiment)

    if not isinstance(experiment, Phase7Experiment):
        raise TypeError(
            "experiment must be a registered experiment ID "
            "or Phase7Experiment."
        )

    registered = get_phase7_experiment(experiment.experiment_id)
    if experiment != registered:
        raise ValueError(
            "Phase 7 experiment differs from the frozen registry."
        )

    return registered


def _validate_oof_experiment(
    oof: Phase7OOFPredictions,
    spec: Phase7Experiment,
) -> None:
    if not isinstance(oof, Phase7OOFPredictions):
        raise TypeError("oof must be Phase7OOFPredictions.")

    if oof.experiment_id != spec.experiment_id:
        raise ValueError(
            "OOF experiment ID does not match threshold experiment."
        )


def optimize_phase7_threshold(
    oof: Phase7OOFPredictions,
    kp_intervals: pd.DataFrame,
    experiment: str | Phase7Experiment,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day: float = PHASE7_MAX_FAR_PER_DAY,
    stability_min_far_per_day: float = DEFAULT_STABILITY_MIN_FAR_PER_DAY,
    progress: bool = False,
) -> Phase7ThresholdOptimization:
    """Recalibrate the operational threshold for one Phase 7 experiment.

    The model, predictors, imbalance strategy, cooldown, threshold grid,
    and FAR constraint remain frozen. Only retrospective truth (T, H)
    and therefore the development-OOF probability distribution may differ.
    """

    spec = _resolve_experiment(experiment)
    _validate_oof_experiment(oof, spec)

    events = build_phase7_events(
        kp_intervals,
        spec,
    )

    result: Phase6ThresholdOptimization = optimize_phase6_threshold(
        oof.table,
        events,
        thresholds=thresholds,
        max_far_per_day=max_far_per_day,
        stability_min_far_per_day=stability_min_far_per_day,
        cooldown_hours=PHASE7_ALERT_COOLDOWN_HOURS,
        horizon_hours=spec.horizon_hours,
        progress=progress,
    )

    return Phase7ThresholdOptimization(
        experiment_id=spec.experiment_id,
        selected_threshold=result.selected_threshold,
        global_threshold_table=result.global_threshold_table,
        fold_threshold_table=result.fold_threshold_table,
        fold_selected_thresholds=result.fold_selected_thresholds,
        stability_thresholds=result.stability_thresholds,
    )