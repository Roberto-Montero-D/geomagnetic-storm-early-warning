"""Fold-aware Phase 6 operational threshold optimization.

This module consumes the frozen Phase 6 OOF prediction table. Alert episodes
are constructed independently inside each validation fold, so no episode can
bridge fold boundaries.

The global operational threshold is the lowest threshold on the frozen
0.01--0.99 grid whose aggregate OOF FAR/day is <= 0.20.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.cross_fold import _events_for_validation
from src.evaluation.oof_predictions import PHASE6_OOF_COLUMNS
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import (
    DEFAULT_MAX_FAR_PER_DAY,
    DEFAULT_THRESHOLD_GRID,
    _validate_threshold_grid,
)


DEFAULT_STABILITY_MIN_FAR_PER_DAY = 0.15

_THRESHOLD_COLUMNS = (
    "threshold",
    "event_recall",
    "false_alarm_rate_per_day",
    "n_events",
    "n_detected_events",
    "n_alert_episodes",
    "n_false_alarm_episodes",
    "valid_exposure_hours",
    "far_feasible",
    "in_stability_region",
)

_FOLD_THRESHOLD_COLUMNS = (
    "fold",
    *_THRESHOLD_COLUMNS,
)


@dataclass(frozen=True)
class Phase6ThresholdOptimization:
    """Complete threshold-selection result for frozen Phase 6 OOF data."""

    selected_threshold: float | None
    global_threshold_table: pd.DataFrame
    fold_threshold_table: pd.DataFrame
    fold_selected_thresholds: dict[str, float | None]
    stability_thresholds: tuple[float, ...]


def _validate_oof_input(oof: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(oof, pd.DataFrame):
        raise TypeError("oof must be a pandas DataFrame.")

    if tuple(oof.columns) != PHASE6_OOF_COLUMNS:
        raise ValueError(
            "oof columns must exactly match the frozen Phase 6 OOF contract."
        )

    if not isinstance(oof.index, pd.DatetimeIndex):
        raise TypeError("oof index must be a pandas DatetimeIndex.")

    if oof.index.name != "timestamp":
        raise ValueError("oof index must be named timestamp.")

    if oof.index.hasnans:
        raise ValueError("oof timestamps must not contain NaT.")

    if oof.index.has_duplicates:
        raise ValueError("oof timestamps must be unique.")

    if not oof.index.is_monotonic_increasing:
        raise ValueError("oof timestamps must be monotonically increasing.")

    if oof.empty:
        raise ValueError("oof must contain at least one prediction row.")

    probability = pd.to_numeric(oof["probability"], errors="raise").astype(float)
    if probability.isna().any():
        raise ValueError("Phase 6 OOF probabilities must not be missing.")
    if not np.isfinite(probability.to_numpy()).all():
        raise ValueError("Phase 6 OOF probabilities must be finite.")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("Phase 6 OOF probabilities must lie in [0, 1].")

    target = pd.to_numeric(oof["target"], errors="raise")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Phase 6 OOF target must be non-missing and binary.")

    if oof["fold"].isna().any():
        raise ValueError("Phase 6 OOF fold labels must not be missing.")

    return oof.copy()


def _validate_far_limits(
    max_far_per_day: float,
    stability_min_far_per_day: float,
) -> tuple[float, float]:
    max_far = float(max_far_per_day)
    stability_min = float(stability_min_far_per_day)

    if not np.isfinite(max_far) or max_far < 0.0:
        raise ValueError(
            "max_far_per_day must be finite and non-negative."
        )

    if not np.isfinite(stability_min) or stability_min < 0.0:
        raise ValueError(
            "stability_min_far_per_day must be finite and non-negative."
        )

    if stability_min > max_far:
        raise ValueError(
            "stability_min_far_per_day cannot exceed max_far_per_day."
        )

    return max_far, stability_min


def _evaluate_fold(
    fold_frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    threshold: float,
    cooldown_hours: int,
    horizon_hours: int,
) -> dict[str, object]:
    probability = pd.Series(
        fold_frame["probability"].to_numpy(dtype=float),
        index=fold_frame.index,
        name="probability",
        dtype=float,
    )

    scoped_events = _events_for_validation(
        events,
        probability.index,
        horizon_hours,
    )

    episodes, metrics = evaluate_operational_series(
        probability,
        scoped_events,
        threshold=threshold,
        cooldown_hours=cooldown_hours,
        horizon_hours=horizon_hours,
    )

    detected_event_ids = set(
        episodes.loc[
            episodes["associated_event_id"].notna(),
            "associated_event_id",
        ].astype(int)
    )

    eligible_event_ids = set(
        pd.to_numeric(
            scoped_events["event_id"],
            errors="raise",
        ).astype(int)
    )

    detected_event_ids &= eligible_event_ids

    return {
        "n_events": len(eligible_event_ids),
        "detected_event_ids": detected_event_ids,
        "n_alert_episodes": metrics.n_alert_episodes,
        "n_false_alarm_episodes": metrics.n_false_alarm_episodes,
        "valid_exposure_hours": int(probability.notna().sum()),
    }


def _aggregate_threshold(
    fold_rows: list[dict[str, object]],
) -> dict[str, object]:
    n_events = sum(int(row["n_events"]) for row in fold_rows)
    n_detected = sum(
        len(row["detected_event_ids"])
        for row in fold_rows
    )
    n_alerts = sum(
        int(row["n_alert_episodes"])
        for row in fold_rows
    )
    n_false = sum(
        int(row["n_false_alarm_episodes"])
        for row in fold_rows
    )
    exposure = sum(
        int(row["valid_exposure_hours"])
        for row in fold_rows
    )

    recall = (
        np.nan
        if n_events == 0
        else n_detected / n_events
    )

    far = (
        np.nan
        if exposure == 0
        else n_false / (exposure / 24.0)
    )

    return {
        "event_recall": float(recall) if not pd.isna(recall) else np.nan,
        "false_alarm_rate_per_day": float(far) if not pd.isna(far) else np.nan,
        "n_events": n_events,
        "n_detected_events": n_detected,
        "n_alert_episodes": n_alerts,
        "n_false_alarm_episodes": n_false,
        "valid_exposure_hours": exposure,
    }


def _row_with_constraints(
    threshold: float,
    aggregate: dict[str, object],
    *,
    max_far: float,
    stability_min: float,
) -> dict[str, object]:
    far = aggregate["false_alarm_rate_per_day"]

    feasible = bool(
        not pd.isna(far)
        and far <= max_far
    )

    stable = bool(
        feasible
        and far >= stability_min
    )

    return {
        "threshold": float(threshold),
        **aggregate,
        "far_feasible": feasible,
        "in_stability_region": stable,
    }


def optimize_phase6_threshold(
    oof: pd.DataFrame,
    events: pd.DataFrame,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day: float = DEFAULT_MAX_FAR_PER_DAY,
    stability_min_far_per_day: float = DEFAULT_STABILITY_MIN_FAR_PER_DAY,
    cooldown_hours: int = 3,
    horizon_hours: int = 6,
    progress: bool = False,
) -> Phase6ThresholdOptimization:
    """Select the frozen global Phase 6 operational threshold.

    Evaluation is fold-aware:
    1. construct alert episodes independently inside each OOF fold;
    2. associate those episodes with events scoped to that validation window;
    3. aggregate event counts, false alarms, and valid exposure;
    4. select the lowest threshold satisfying FAR/day <= max_far_per_day.

    Fold-specific selected thresholds are diagnostics only. They do not replace
    the single globally selected Phase 6 operational threshold.
    """

    frame = _validate_oof_input(oof)
    threshold_grid = _validate_threshold_grid(thresholds)
    max_far, stability_min = _validate_far_limits(
        max_far_per_day,
        stability_min_far_per_day,
    )

    fold_names = list(dict.fromkeys(frame["fold"].astype(str)))
    if not fold_names:
        raise ValueError("Phase 6 OOF must contain at least one fold.")

    global_rows: list[dict[str, object]] = []
    fold_rows_output: list[dict[str, object]] = []

    total = len(threshold_grid)

    for number, threshold in enumerate(threshold_grid, start=1):
        if progress:
            print(
                f"    threshold {number:02d}/{total} "
                f"(tau={threshold:.2f})",
                flush=True,
            )

        evaluated_folds: list[dict[str, object]] = []

        for fold_name in fold_names:
            fold_frame = frame.loc[
                frame["fold"].astype(str) == fold_name
            ].copy()

            fold_result = _evaluate_fold(
                fold_frame,
                events,
                threshold=threshold,
                cooldown_hours=cooldown_hours,
                horizon_hours=horizon_hours,
            )
            evaluated_folds.append(fold_result)

            fold_aggregate = _aggregate_threshold([fold_result])
            fold_rows_output.append(
                {
                    "fold": fold_name,
                    **_row_with_constraints(
                        threshold,
                        fold_aggregate,
                        max_far=max_far,
                        stability_min=stability_min,
                    ),
                }
            )

        aggregate = _aggregate_threshold(evaluated_folds)
        global_rows.append(
            _row_with_constraints(
                threshold,
                aggregate,
                max_far=max_far,
                stability_min=stability_min,
            )
        )

    global_table = pd.DataFrame(
        global_rows,
        columns=_THRESHOLD_COLUMNS,
    )

    fold_table = pd.DataFrame(
        fold_rows_output,
        columns=_FOLD_THRESHOLD_COLUMNS,
    )

    feasible = global_table.loc[
        global_table["far_feasible"],
        "threshold",
    ]

    selected_threshold = (
        None
        if feasible.empty
        else float(feasible.iloc[0])
    )

    fold_selected: dict[str, float | None] = {}
    for fold_name in fold_names:
        rows = fold_table.loc[
            (fold_table["fold"] == fold_name)
            & fold_table["far_feasible"]
        ]
        fold_selected[fold_name] = (
            None
            if rows.empty
            else float(rows.iloc[0]["threshold"])
        )

    stability_thresholds = tuple(
        float(value)
        for value in global_table.loc[
            global_table["in_stability_region"],
            "threshold",
        ]
    )

    return Phase6ThresholdOptimization(
        selected_threshold=selected_threshold,
        global_threshold_table=global_table,
        fold_threshold_table=fold_table,
        fold_selected_thresholds=fold_selected,
        stability_thresholds=stability_thresholds,
    )
