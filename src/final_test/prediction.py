"""Outcome-blind Phase 8 Final Test prediction generation.

This module may fit the one frozen Phase 8 model and generate probabilities for
the protected Final Test feature matrix. It does not accept Final Test targets,
does not construct alerts, does not calculate metrics, and does not perform
threshold selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.final_test.contract import (
    PHASE8_MODEL_CONFIG_ID,
    PHASE8_OPERATIONAL_THRESHOLD,
)
from src.final_test.materialization import Phase8Materialization
from src.model_selection.factories import make_phase5_model_by_id


PHASE8_PREDICTION_COLUMNS = ("probability",)


@dataclass(frozen=True)
class Phase8Predictions:
    """Protected prediction artifact with no outcome columns."""

    config_id: str
    operational_threshold: float
    table: pd.DataFrame


def _validate_prediction_table(
    table: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
) -> None:
    if tuple(table.columns) != PHASE8_PREDICTION_COLUMNS:
        raise AssertionError(
            "Phase 8 prediction artifact exposed non-probability columns."
        )
    if not table.index.equals(expected_index):
        raise AssertionError(
            "Phase 8 prediction timestamps changed during prediction."
        )
    if table.index.has_duplicates:
        raise AssertionError(
            "Phase 8 prediction timestamps must be unique."
        )

    probability = pd.to_numeric(
        table["probability"],
        errors="raise",
    )
    if probability.isna().any():
        raise AssertionError(
            "Phase 8 predictions contain missing probabilities."
        )
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise AssertionError(
            "Phase 8 probabilities must lie in [0, 1]."
        )


def generate_phase8_predictions(
    materialized: Phase8Materialization,
    *,
    progress: bool = False,
) -> Phase8Predictions:
    """Fit exactly one frozen model and predict the protected feature matrix."""

    if not isinstance(materialized, Phase8Materialization):
        raise TypeError(
            "materialized must be a Phase8Materialization."
        )

    model = make_phase5_model_by_id(PHASE8_MODEL_CONFIG_ID)

    if progress:
        print(
            "Phase 8 frozen fit: "
            f"{len(materialized.x_train):,} rows; "
            f"predict {len(materialized.x_final_test):,} rows"
        )

    model.fit(
        materialized.x_train,
        materialized.y_train,
    )

    raw = np.asarray(
        model.predict_proba(materialized.x_final_test),
        dtype=float,
    )

    if raw.ndim != 2 or raw.shape != (
        len(materialized.x_final_test),
        2,
    ):
        raise AssertionError(
            "Frozen Phase 8 classifier must return two-class probabilities."
        )

    table = pd.DataFrame(
        {"probability": raw[:, 1]},
        index=materialized.x_final_test.index.copy(),
    )
    table.index.name = materialized.x_final_test.index.name

    _validate_prediction_table(
        table,
        pd.DatetimeIndex(materialized.x_final_test.index),
    )

    return Phase8Predictions(
        config_id=PHASE8_MODEL_CONFIG_ID,
        operational_threshold=PHASE8_OPERATIONAL_THRESHOLD,
        table=table,
    )
