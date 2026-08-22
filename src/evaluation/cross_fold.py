"""Cross-fold Phase 2 operational evaluation.

Validation folds remain explicit. Alert episodes are never constructed across
the multi-year gaps between validation windows.

Phase 2 probabilistic thresholds are baseline-evaluation thresholds only.
They do not replace the final global OOF operational threshold procedure
reserved for Phase 6.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import PERIOD_FINAL_TEST
from src.evaluation.development_predictions import (
    BASELINE_NAMES,
    generate_fold_baseline_predictions,
)
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import (
    DEFAULT_MAX_FAR_PER_DAY,
    DEFAULT_THRESHOLD_GRID,
    _validate_threshold_grid,
)

DETERMINISTIC_BASELINES = (
    "B0_persistence",
    "B1_physical",
)

PROBABILISTIC_BASELINES = (
    "B2_logistic",
    "B3_extratrees",
)

DETERMINISTIC_THRESHOLD = 0.5


@dataclass(frozen=True)
class CrossFoldEvaluation:
    selected_thresholds: dict[str, float | None]
    fold_metrics: pd.DataFrame
    threshold_tables: dict[str, pd.DataFrame]


def _progress(
    enabled: bool,
    message: str,
) -> None:
    """Print observational progress without changing evaluation logic."""
    if enabled:
        print(message, flush=True)


def _events_for_validation(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    horizon_hours: int,
) -> pd.DataFrame:
    if len(index) == 0:
        return events.iloc[0:0].copy()

    start = index.min()
    end = index.max()

    starts = pd.to_datetime(
        events["start_time"],
        errors="raise",
    )

    mask = (
        (starts >= start)
        & (starts <= end)
    )

    return events.loc[mask].copy()


def _validate_development_folds(
    folds: tuple[DevelopmentFold, ...]
    | list[DevelopmentFold],
    splits: pd.DataFrame,
) -> None:
    """Reject any development fold touching the protected Final Test."""

    for fold in folds:
        for role, index in (
            ("train", fold.train_index),
            ("validation", fold.validation_index),
        ):
            if not index.isin(splits.index).all():
                raise ValueError(
                    f"{fold.name} {role} index contains "
                    "timestamps absent from splits."
                )

            periods = splits.loc[
                index,
                "period",
            ]

            if (
                periods
                == PERIOD_FINAL_TEST
            ).any():
                raise ValueError(
                    f"{fold.name} {role} index touches "
                    "the protected Final Test."
                )


def _aggregate_far(
    fold_rows: list[dict],
) -> float:
    exposure = sum(
        row["valid_exposure_hours"]
        for row in fold_rows
    )

    false_alarms = sum(
        row["n_false_alarm_episodes"]
        for row in fold_rows
    )

    if exposure == 0:
        return np.nan

    return (
        false_alarms
        / (exposure / 24.0)
    )


def _select_cross_fold_threshold(
    predictions,
    folds,
    events,
    baseline: str,
    *,
    thresholds,
    max_far_per_day: float,
    cooldown_hours: int,
    horizon_hours: int,
    progress: bool = False,
) -> tuple[
    float | None,
    pd.DataFrame,
]:
    """Select minimum feasible Phase 2 baseline-evaluation threshold."""

    thresholds = _validate_threshold_grid(
        thresholds
    )
    total_thresholds = len(
        thresholds
    )

    max_far = float(
        max_far_per_day
    )

    if (
        not np.isfinite(max_far)
        or max_far < 0
    ):
        raise ValueError(
            "max_far_per_day must be finite "
            "and non-negative."
        )

    threshold_rows = []

    for (
        threshold_number,
        threshold,
    ) in enumerate(
        thresholds,
        start=1,
    ):
        _progress(
            progress,
            (
                f"      {baseline}: threshold "
                f"{threshold_number:02d}/"
                f"{total_thresholds} "
                f"(tau={threshold:.2f})"
            ),
        )

        fold_rows = []

        for (
            fold,
            prediction,
        ) in zip(
            folds,
            predictions,
        ):
            probability = (
                prediction
                .probabilities[
                    baseline
                ]
            )

            scoped_events = (
                _events_for_validation(
                    events,
                    probability.index,
                    horizon_hours,
                )
            )

            _, metrics = (
                evaluate_operational_series(
                    probability,
                    scoped_events,
                    threshold=threshold,
                    cooldown_hours=(
                        cooldown_hours
                    ),
                    horizon_hours=(
                        horizon_hours
                    ),
                )
            )

            fold_rows.append(
                {
                    "valid_exposure_hours":
                        int(
                            probability
                            .notna()
                            .sum()
                        ),
                    "n_false_alarm_episodes":
                        metrics
                        .n_false_alarm_episodes,
                }
            )

        far = _aggregate_far(
            fold_rows
        )

        threshold_rows.append(
            {
                "threshold":
                    threshold,
                "false_alarm_rate_per_day":
                    far,
                "far_feasible":
                    bool(
                        not pd.isna(far)
                        and (
                            far
                            <= max_far
                        )
                    ),
            }
        )

    table = pd.DataFrame(
        threshold_rows
    )

    feasible = table.loc[
        table["far_feasible"],
        "threshold",
    ]

    selected = (
        None
        if feasible.empty
        else float(
            feasible.iloc[0]
        )
    )

    return selected, table


def evaluate_development_folds(
    dataset: pd.DataFrame,
    folds: tuple[
        DevelopmentFold,
        ...,
    ]
    | list[DevelopmentFold],
    events: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day: float = (
        DEFAULT_MAX_FAR_PER_DAY
    ),
    cooldown_hours: int = 3,
    horizon_hours: int = 6,
    progress: bool = False,
) -> CrossFoldEvaluation:
    """Generate and evaluate B0-B3 across protected development folds."""

    _validate_development_folds(
        folds,
        splits,
    )

    _progress(
        progress,
        "    Generating protected fold predictions...",
    )

    predictions = [
        generate_fold_baseline_predictions(
            dataset,
            fold,
        )
        for fold in folds
    ]

    selected = {
        name:
            DETERMINISTIC_THRESHOLD
        for name
        in DETERMINISTIC_BASELINES
    }

    threshold_tables = {}

    for baseline in (
        PROBABILISTIC_BASELINES
    ):
        _progress(
            progress,
            (
                "    Selecting development "
                f"threshold for {baseline}..."
            ),
        )

        (
            threshold,
            table,
        ) = (
            _select_cross_fold_threshold(
                predictions,
                folds,
                events,
                baseline,
                thresholds=thresholds,
                max_far_per_day=(
                    max_far_per_day
                ),
                cooldown_hours=(
                    cooldown_hours
                ),
                horizon_hours=(
                    horizon_hours
                ),
                progress=progress,
            )
        )

        selected[
            baseline
        ] = threshold

        threshold_tables[
            baseline
        ] = table

    _progress(
        progress,
        (
            "    Computing selected-threshold "
            "fold metrics..."
        ),
    )

    metric_rows = []

    for (
        fold,
        prediction,
    ) in zip(
        folds,
        predictions,
    ):
        for baseline in (
            BASELINE_NAMES
        ):
            threshold = (
                selected[
                    baseline
                ]
            )

            probability = (
                prediction
                .probabilities[
                    baseline
                ]
            )

            exposure = int(
                probability
                .notna()
                .sum()
            )

            if threshold is None:
                metric_rows.append(
                    {
                        "fold":
                            fold.name,
                        "baseline":
                            baseline,
                        "threshold":
                            np.nan,
                        "event_recall":
                            np.nan,
                        "false_alarm_rate_per_day":
                            np.nan,
                        "median_lead_time":
                            pd.NaT,
                        "n_alert_episodes":
                            0,
                        "n_false_alarm_episodes":
                            0,
                        "n_early_detections":
                            0,
                        "n_late_detections":
                            0,
                        "valid_exposure_hours":
                            exposure,
                    }
                )
                continue

            scoped_events = (
                _events_for_validation(
                    events,
                    probability.index,
                    horizon_hours,
                )
            )

            _, metrics = (
                evaluate_operational_series(
                    probability,
                    scoped_events,
                    threshold=threshold,
                    cooldown_hours=(
                        cooldown_hours
                    ),
                    horizon_hours=(
                        horizon_hours
                    ),
                )
            )

            metric_rows.append(
                {
                    "fold":
                        fold.name,
                    "baseline":
                        baseline,
                    "threshold":
                        threshold,
                    "event_recall":
                        metrics
                        .event_recall,
                    "false_alarm_rate_per_day":
                        metrics
                        .false_alarm_rate_per_day,
                    "median_lead_time":
                        metrics
                        .median_lead_time,
                    "n_alert_episodes":
                        metrics
                        .n_alert_episodes,
                    "n_false_alarm_episodes":
                        metrics
                        .n_false_alarm_episodes,
                    "n_early_detections":
                        metrics
                        .n_early_detections,
                    "n_late_detections":
                        metrics
                        .n_late_detections,
                    "valid_exposure_hours":
                        exposure,
                }
            )

    return CrossFoldEvaluation(
        selected,
        pd.DataFrame(
            metric_rows
        ),
        threshold_tables,
    )