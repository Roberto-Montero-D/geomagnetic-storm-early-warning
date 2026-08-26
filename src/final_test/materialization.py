"""Phase 8 Final Test materialization without outcome exposure.

This module builds the frozen pre-2022 training sample and the protected
2022-2025 prediction feature matrix. It deliberately does NOT return Final Test
targets, event truth, alerts, or metrics.

Training rows:
- prediction_time in [1996-01-01, 2022-01-01)
- supervised_eligible == True

Prediction rows:
- period == final_test
- selected Phase 8 features are complete

Final Test target-known status is NOT used to decide which rows receive model
predictions. Outcome eligibility is deferred to the later one-time scoring
layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.dataset.temporal_splits import PERIOD_FINAL_TEST
from src.final_test.contract import (
    PHASE8_FEATURES,
    PHASE8_FINAL_TEST_END_EXCLUSIVE,
    PHASE8_FINAL_TEST_START,
    PHASE8_TRAIN_END_EXCLUSIVE,
    PHASE8_TRAIN_START,
)


@dataclass(frozen=True)
class Phase8Materialization:
    """Frozen training data and outcome-blind Final Test predictors."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_final_test: pd.DataFrame

    @property
    def train_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.x_train.index)

    @property
    def final_test_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.x_final_test.index)


def _validate_aligned(
    dataset: pd.DataFrame,
    status: pd.DataFrame,
    splits: pd.DataFrame,
) -> None:
    if not dataset.index.equals(status.index):
        raise ValueError("dataset and status indices must match exactly.")
    if not dataset.index.equals(splits.index):
        raise ValueError("dataset and splits indices must match exactly.")

    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise TypeError("dataset index must be a DatetimeIndex.")
    if dataset.index.has_duplicates:
        raise ValueError("dataset index must be unique.")
    if not dataset.index.is_monotonic_increasing:
        raise ValueError("dataset index must be chronologically ordered.")

    if "target" not in dataset.columns:
        raise ValueError("dataset must contain target.")
    if "supervised_eligible" not in status.columns:
        raise ValueError("status must contain supervised_eligible.")
    if "features_complete" not in status.columns:
        raise ValueError("status must contain features_complete.")
    if "period" not in splits.columns:
        raise ValueError("splits must contain period.")

    missing = [
        feature
        for feature in PHASE8_FEATURES
        if feature not in dataset.columns
    ]
    if missing:
        raise ValueError(
            "dataset is missing frozen Phase 8 features: "
            f"{missing}"
        )


def materialize_phase8_data(
    dataset: pd.DataFrame,
    status: pd.DataFrame,
    splits: pd.DataFrame,
) -> Phase8Materialization:
    """Build the frozen Phase 8 fit/predict matrices without test outcomes."""

    _validate_aligned(dataset, status, splits)

    index = dataset.index
    features = list(PHASE8_FEATURES)

    train_calendar = (
        (index >= PHASE8_TRAIN_START)
        & (index < PHASE8_TRAIN_END_EXCLUSIVE)
    )
    train_mask = (
        train_calendar
        & status["supervised_eligible"].astype(bool).to_numpy()
    )

    final_calendar = (
        (index >= PHASE8_FINAL_TEST_START)
        & (index < PHASE8_FINAL_TEST_END_EXCLUSIVE)
    )
    final_period = splits["period"].eq(PERIOD_FINAL_TEST).to_numpy()

    if not (final_calendar == final_period).all():
        raise AssertionError(
            "Phase 8 Final Test calendar disagrees with Phase 1 split contract."
        )

    # Prediction eligibility is outcome-blind: use feature completeness only.
    final_mask = (
        final_period
        & status["features_complete"].astype(bool).to_numpy()
    )

    x_train = dataset.loc[train_mask, features].copy()
    y_train = dataset.loc[train_mask, "target"].copy()
    x_final = dataset.loc[final_mask, features].copy()

    if x_train.empty:
        raise ValueError("Phase 8 training sample is empty.")
    if x_final.empty:
        raise ValueError("Phase 8 Final Test prediction sample is empty.")

    if y_train.isna().any():
        raise AssertionError(
            "Phase 8 training sample contains unknown targets."
        )
    if x_train.isna().any().any():
        raise AssertionError(
            "Phase 8 training predictors contain missing values."
        )
    if x_final.isna().any().any():
        raise AssertionError(
            "Phase 8 Final Test predictors contain missing values."
        )

    if x_train.index.max() >= PHASE8_FINAL_TEST_START:
        raise AssertionError(
            "Protected Final Test entered Phase 8 training data."
        )
    if x_final.index.min() < PHASE8_FINAL_TEST_START:
        raise AssertionError(
            "Development rows entered Phase 8 Final Test predictors."
        )
    if x_final.index.max() >= PHASE8_FINAL_TEST_END_EXCLUSIVE:
        raise AssertionError(
            "Rows after the protected interval entered Phase 8 predictors."
        )

    if tuple(x_train.columns) != PHASE8_FEATURES:
        raise AssertionError("Phase 8 training feature order drifted.")
    if tuple(x_final.columns) != PHASE8_FEATURES:
        raise AssertionError("Phase 8 Final Test feature order drifted.")

    return Phase8Materialization(
        x_train=x_train,
        y_train=y_train,
        x_final_test=x_final,
    )
