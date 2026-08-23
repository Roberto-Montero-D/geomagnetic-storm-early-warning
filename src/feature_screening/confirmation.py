"""Frozen Phase 3 walk-forward confirmation for A/E/C."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.baselines.extratrees import make_extratrees_model
from src.baselines.framework import DevelopmentFold, get_development_xy
from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    PERIOD_INITIAL_TRAIN,
    PERIOD_VALIDATION_1,
    PERIOD_VALIDATION_2,
    PERIOD_VALIDATION_3,
)
from src.evaluation.operational import evaluate_operational_series
from src.evaluation.threshold_selection import (
    DEFAULT_MAX_FAR_PER_DAY,
    DEFAULT_THRESHOLD_GRID,
)
from src.feature_screening.manifests import (
    PHASE3_EXTRATREES_PARAMS,
    PHASE3_FEATURE_SETS,
)
from src.feature_screening.screening import (
    _events_for_validation,
    _select_single_fold_threshold,
)


PHASE3_ADVANCING_EXPERIMENTS = ("A", "E", "C")

PHASE3_CONFIRMATION_FOLDS = (
    "walk_forward_1",
    "walk_forward_2",
)

PHASE3_CONFIRMATION_PERIODS = {
    "walk_forward_1": {
        "train": (
            PERIOD_INITIAL_TRAIN,
            PERIOD_VALIDATION_1,
        ),
        "validation": (
            PERIOD_VALIDATION_2,
        ),
    },
    "walk_forward_2": {
        "train": (
            PERIOD_INITIAL_TRAIN,
            PERIOD_VALIDATION_1,
            PERIOD_VALIDATION_2,
        ),
        "validation": (
            PERIOD_VALIDATION_3,
        ),
    },
}


@dataclass(frozen=True)
class ConfirmationFoldResult:
    experiment: str
    fold: str
    threshold: float | None
    event_recall: float
    false_alarm_rate_per_day: float
    pr_auc: float
    operationally_feasible: bool
    threshold_table: pd.DataFrame


@dataclass(frozen=True)
class Phase3ConfirmationResult:
    fold_results: dict[tuple[str, str], ConfirmationFoldResult]
    ranking: pd.DataFrame
    selected_experiment: str | None


def _validate_confirmation_fold(
    fold: DevelopmentFold,
    splits: pd.DataFrame,
) -> None:
    """Validate a Phase 3 confirmation fold against frozen temporal periods."""

    if fold.name not in PHASE3_CONFIRMATION_PERIODS:
        raise ValueError(
            "Confirmation requires walk_forward_1 or walk_forward_2."
        )

    for role, index in (
        ("train", fold.train_index),
        ("validation", fold.validation_index),
    ):
        if not index.isin(splits.index).all():
            raise ValueError(
                f"{fold.name} {role} contains timestamps absent from splits."
            )

        periods = splits.loc[index, "period"]

        if (periods == PERIOD_FINAL_TEST).any():
            raise ValueError(
                f"{fold.name} {role} touches the protected Final Test."
            )

        allowed = PHASE3_CONFIRMATION_PERIODS[fold.name][role]

        if not periods.isin(allowed).all():
            raise ValueError(
                f"{fold.name} {role} contains rows outside its frozen "
                f"atomic periods: {allowed}."
            )

    if len(
        fold.train_index.intersection(fold.validation_index)
    ) != 0:
        raise ValueError(
            f"{fold.name} train and validation indices overlap."
        )

    if (
        len(fold.train_index) > 0
        and len(fold.validation_index) > 0
        and fold.train_index.max() >= fold.validation_index.min()
    ):
        raise ValueError(
            f"{fold.name} violates chronological "
            "train-before-validation order."
        )


def _fit_candidate(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
    experiment: str,
) -> tuple[pd.Series, pd.Series]:
    """Fit one frozen candidate on one confirmation fold."""

    features = PHASE3_FEATURE_SETS[experiment]

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) = get_development_xy(
        dataset,
        fold,
        features,
    )

    if x_train.isna().any().any():
        raise AssertionError(
            "Phase 3 confirmation training predictors "
            "contain missing values."
        )

    if x_validation.isna().any().any():
        raise AssertionError(
            "Phase 3 confirmation validation predictors "
            "contain missing values."
        )

    if len(np.unique(y_train.to_numpy())) != 2:
        raise ValueError(
            "Phase 3 confirmation training target "
            "must contain both classes."
        )

    model = make_extratrees_model(
        n_estimators=PHASE3_EXTRATREES_PARAMS["n_estimators"],
        max_depth=PHASE3_EXTRATREES_PARAMS["max_depth"],
        random_state=PHASE3_EXTRATREES_PARAMS["random_state"],
    )

    model.fit(
        x_train,
        y_train.astype(int),
    )

    probability = pd.Series(
        model.predict_proba(x_validation)[:, 1],
        index=x_validation.index,
        name="probability",
        dtype=float,
    )

    return probability, y_validation.astype(int)


def evaluate_confirmation_fold(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
    events: pd.DataFrame,
    splits: pd.DataFrame,
    experiment: str,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    max_far_per_day: float = DEFAULT_MAX_FAR_PER_DAY,
    progress: bool = False,
) -> ConfirmationFoldResult:
    """Evaluate one frozen candidate on one confirmation fold."""

    if experiment not in PHASE3_ADVANCING_EXPERIMENTS:
        raise ValueError(
            "Only frozen Phase 3 candidates A, E, and C may be confirmed."
        )

    _validate_confirmation_fold(
        fold,
        splits,
    )

    if progress:
        print(
            f"    {fold.name}: fitting {experiment} "
            f"({len(PHASE3_FEATURE_SETS[experiment])} features)...",
            flush=True,
        )

    probability, y_validation = _fit_candidate(
        dataset,
        fold,
        experiment,
    )

    scoped_events = _events_for_validation(
        events,
        probability.index,
    )

    pr_auc = float(
        average_precision_score(
            y_validation.to_numpy(),
            probability.to_numpy(),
        )
    )

    threshold, threshold_table = _select_single_fold_threshold(
        probability,
        scoped_events,
        thresholds=thresholds,
        max_far_per_day=max_far_per_day,
        cooldown_hours=3,
        horizon_hours=6,
        progress=progress,
        experiment=f"{fold.name}/{experiment}",
    )

    if threshold is None:
        return ConfirmationFoldResult(
            experiment=experiment,
            fold=fold.name,
            threshold=None,
            event_recall=np.nan,
            false_alarm_rate_per_day=np.nan,
            pr_auc=pr_auc,
            operationally_feasible=False,
            threshold_table=threshold_table,
        )

    _, metrics = evaluate_operational_series(
        probability,
        scoped_events,
        threshold=threshold,
        cooldown_hours=3,
        horizon_hours=6,
    )

    return ConfirmationFoldResult(
        experiment=experiment,
        fold=fold.name,
        threshold=threshold,
        event_recall=metrics.event_recall,
        false_alarm_rate_per_day=(
            metrics.false_alarm_rate_per_day
        ),
        pr_auc=pr_auc,
        operationally_feasible=True,
        threshold_table=threshold_table,
    )


def rank_confirmation_candidates(
    fold_results: dict[
        tuple[str, str],
        ConfirmationFoldResult,
    ],
) -> tuple[pd.DataFrame, str | None]:
    """Rank candidates using the frozen Phase 3 confirmation rule."""

    rows = []

    order = {
        name: index
        for index, name in enumerate(
            PHASE3_ADVANCING_EXPERIMENTS
        )
    }

    for experiment in PHASE3_ADVANCING_EXPERIMENTS:
        items = [
            fold_results[(experiment, fold_name)]
            for fold_name in PHASE3_CONFIRMATION_FOLDS
        ]

        feasible = all(
            item.operationally_feasible
            for item in items
        )

        recalls = [
            item.event_recall
            for item in items
        ]

        pr_aucs = [
            item.pr_auc
            for item in items
        ]

        fars = [
            item.false_alarm_rate_per_day
            for item in items
        ]

        rows.append(
            {
                "experiment": experiment,
                "confirmation_feasible": feasible,
                "minimum_event_recall": (
                    min(recalls)
                    if feasible
                    else np.nan
                ),
                "mean_event_recall": (
                    float(np.mean(recalls))
                    if feasible
                    else np.nan
                ),
                "mean_pr_auc": float(
                    np.mean(pr_aucs)
                ),
                "mean_false_alarm_rate_per_day": (
                    float(np.mean(fars))
                    if feasible
                    else np.nan
                ),
                "_order": order[experiment],
            }
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "confirmation_feasible",
                "minimum_event_recall",
                "mean_event_recall",
                "mean_pr_auc",
                "mean_false_alarm_rate_per_day",
                "_order",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            na_position="last",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    feasible = ranking.loc[
        ranking["confirmation_feasible"],
        "experiment",
    ]

    selected = (
        None
        if feasible.empty
        else str(feasible.iloc[0])
    )

    return (
        ranking.drop(columns="_order"),
        selected,
    )


def evaluate_phase3_confirmation(
    dataset: pd.DataFrame,
    fold_map: dict[str, DevelopmentFold],
    events: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    progress: bool = False,
) -> Phase3ConfirmationResult:
    """Evaluate A/E/C on both frozen confirmation folds."""

    missing = [
        fold_name
        for fold_name in PHASE3_CONFIRMATION_FOLDS
        if fold_name not in fold_map
    ]

    if missing:
        raise ValueError(
            f"Missing confirmation folds: {missing}"
        )

    results: dict[
        tuple[str, str],
        ConfirmationFoldResult,
    ] = {}

    for fold_name in PHASE3_CONFIRMATION_FOLDS:
        for experiment in PHASE3_ADVANCING_EXPERIMENTS:
            results[(experiment, fold_name)] = (
                evaluate_confirmation_fold(
                    dataset,
                    fold_map[fold_name],
                    events,
                    splits,
                    experiment,
                    progress=progress,
                )
            )

    ranking, selected = rank_confirmation_candidates(
        results
    )

    return Phase3ConfirmationResult(
        fold_results=results,
        ranking=ranking,
        selected_experiment=selected,
    )