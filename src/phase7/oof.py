"""Phase 7 experiment-specific OOF prediction generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.baselines.framework import DevelopmentFold, get_development_xy
from src.dataset.builder import TARGET_COLUMN
from src.evaluation.oof_predictions import (
    PHASE6_OOF_COLUMNS,
    _storm_id_for_predictions,
    _validate_oof_table,
)
from src.features.integrated import PRIMARY_FEATURE_COLUMNS
from src.model_selection.confirmation import (
    PHASE5_CONFIRMATION_FOLDS,
    _validate_confirmation_fold,
)
from src.model_selection.factories import make_phase5_model_by_id
from src.phase7.contract import (
    PHASE7_FEATURES,
    PHASE7_MODEL_CONFIG_ID,
    PHASE7_PRIMARY_CONTROL_ID,
    Phase7Experiment,
    build_phase7_events,
    build_phase7_target,
    get_phase7_experiment,
)

PHASE7_OOF_COLUMNS = PHASE6_OOF_COLUMNS


@dataclass(frozen=True)
class Phase7OOFPredictions:
    experiment_id: str
    config_id: str
    table: pd.DataFrame


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
            "Phase 7 experiment specification differs from "
            "the frozen registry."
        )
    return registered


def build_phase7_experiment_dataset(
    base_dataset: pd.DataFrame,
    kp_intervals: pd.DataFrame,
    experiment: str | Phase7Experiment,
) -> pd.DataFrame:
    """Replace only target truth while preserving the canonical features."""

    spec = _resolve_experiment(experiment)

    required_columns = (
        *PRIMARY_FEATURE_COLUMNS,
        TARGET_COLUMN,
    )
    if tuple(base_dataset.columns) != required_columns:
        raise ValueError(
            "base_dataset must contain exactly the canonical "
            "93 features plus target in frozen order."
        )

    if not isinstance(base_dataset.index, pd.DatetimeIndex):
        raise ValueError("base_dataset index must be a DatetimeIndex.")
    if base_dataset.index.has_duplicates:
        raise ValueError(
            "base_dataset prediction timestamps must be unique."
        )

    target = build_phase7_target(
        kp_intervals,
        base_dataset.index,
        spec,
    )

    if not target.index.equals(base_dataset.index):
        raise AssertionError(
            "Phase 7 target did not preserve the canonical "
            "prediction-time index."
        )

    result = base_dataset.copy()
    original_features = result.loc[
        :, list(PRIMARY_FEATURE_COLUMNS)
    ].copy()

    result[TARGET_COLUMN] = target

    if not result.loc[
        :, list(PRIMARY_FEATURE_COLUMNS)
    ].equals(original_features):
        raise AssertionError(
            "Phase 7 target replacement altered predictor values."
        )

    if tuple(result.columns) != required_columns:
        raise AssertionError("Phase 7 dataset column order drifted.")

    return result


def _generate_phase7_oof_from_dataset(
    dataset: pd.DataFrame,
    folds: dict[str, DevelopmentFold],
    events: pd.DataFrame,
    splits: pd.DataFrame,
    spec: Phase7Experiment,
    *,
    progress: bool = False,
) -> Phase7OOFPredictions:
    """Generate OOF predictions from one experiment-specific dataset."""

    parts: list[pd.DataFrame] = []

    for fold_name in PHASE5_CONFIRMATION_FOLDS:
        if fold_name not in folds:
            raise AssertionError(
                f"Missing required Phase 7 fold: {fold_name}"
            )

        fold = folds[fold_name]
        _validate_confirmation_fold(fold, splits, fold_name)

        (
            x_train,
            y_train,
            x_validation,
            y_validation,
        ) = get_development_xy(
            dataset,
            fold,
            PHASE7_FEATURES,
        )

        if (
            x_train.isna().any().any()
            or x_validation.isna().any().any()
        ):
            raise AssertionError(
                "Phase 7 predictors contain missing values."
            )

        if progress:
            print(
                "    "
                f"{spec.experiment_id} / {fold_name} / "
                f"{PHASE7_MODEL_CONFIG_ID}: "
                f"fit {len(x_train):,} rows -> "
                f"predict {len(x_validation):,} rows",
                flush=True,
            )

        model = make_phase5_model_by_id(PHASE7_MODEL_CONFIG_ID)
        model.fit(x_train, y_train.astype(int))

        probability = pd.Series(
            model.predict_proba(x_validation)[:, 1],
            index=x_validation.index,
            name="probability",
            dtype=float,
        )

        part = pd.DataFrame(
            {
                "probability": probability,
                "target": y_validation.astype(int),
                "storm_id": _storm_id_for_predictions(
                    x_validation.index,
                    events,
                ),
                "fold": fold_name,
            },
            index=x_validation.index,
        )
        part.index = pd.DatetimeIndex(
            part.index,
            name="timestamp",
        )
        parts.append(part)

    table = pd.concat(parts, axis=0).sort_index(
        kind="mergesort"
    )
    table = table.loc[:, list(PHASE7_OOF_COLUMNS)]

    _validate_oof_table(table, folds, splits)

    return Phase7OOFPredictions(
        experiment_id=spec.experiment_id,
        config_id=PHASE7_MODEL_CONFIG_ID,
        table=table,
    )


def generate_phase7_oof_predictions(
    base_dataset: pd.DataFrame,
    kp_intervals: pd.DataFrame,
    folds: dict[str, DevelopmentFold],
    splits: pd.DataFrame,
    experiment: str | Phase7Experiment,
    *,
    progress: bool = False,
) -> Phase7OOFPredictions:
    """Generate development-only OOF predictions for one frozen experiment."""

    spec = _resolve_experiment(experiment)
    dataset = build_phase7_experiment_dataset(
        base_dataset,
        kp_intervals,
        spec,
    )
    events = build_phase7_events(
        kp_intervals,
        spec,
    )

    return _generate_phase7_oof_from_dataset(
        dataset,
        folds,
        events,
        splits,
        spec,
        progress=progress,
    )


def assert_phase7_primary_control_dataset(
    base_dataset: pd.DataFrame,
    kp_intervals: pd.DataFrame,
) -> None:
    """Require t5_h6 truth to reproduce the canonical Phase 6 dataset."""

    control_dataset = build_phase7_experiment_dataset(
        base_dataset,
        kp_intervals,
        PHASE7_PRIMARY_CONTROL_ID,
    )

    base_target = pd.to_numeric(
        base_dataset[TARGET_COLUMN],
        errors="coerce",
    )
    control_target = pd.to_numeric(
        control_dataset[TARGET_COLUMN],
        errors="coerce",
    )

    equal = (
        base_target.eq(control_target)
        | (base_target.isna() & control_target.isna())
    )

    if not equal.all():
        raise AssertionError(
            "Phase 7 t5_h6 target does not reproduce the "
            "canonical Phase 6 target. "
            f"Mismatched rows: {int((~equal).sum())}."
        )


def assert_phase7_oof_is_development_only(
    result: Phase7OOFPredictions,
    splits: pd.DataFrame,
) -> None:
    """Explicit Phase 7 Final Test firewall assertion."""

    if not result.table.index.isin(splits.index).all():
        raise AssertionError(
            "Phase 7 OOF contains timestamps outside temporal splits."
        )

    periods = splits.loc[
        result.table.index,
        "period",
    ].astype(str)

    if periods.eq("final_test").any():
        raise AssertionError(
            "Protected Final Test entered Phase 7 OOF predictions."
        )

    probability = pd.to_numeric(
        result.table["probability"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(probability).all():
        raise AssertionError(
            "Phase 7 OOF probabilities must be finite."
        )
