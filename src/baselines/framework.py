"""Protected development-fold framework for Phase 2 baselines.

This module centralizes the rules shared by every baseline:
- use the frozen Phase 1 temporal folds;
- use only supervised-eligible rows;
- keep train and validation strictly chronological;
- never expose protected Final Test rows through the development API.

It does not fit a model or choose a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    development_fold_masks,
)


DEVELOPMENT_FOLD_NAMES = (
    "screening",
    "walk_forward_1",
    "walk_forward_2",
)


@dataclass(frozen=True)
class DevelopmentFold:
    """Materialized supervised train/validation indices for one frozen fold."""

    name: str
    train_index: pd.DatetimeIndex
    validation_index: pd.DatetimeIndex


def _validate_aligned(
    dataset: pd.DataFrame,
    status: pd.DataFrame,
    splits: pd.DataFrame,
) -> None:
    if not dataset.index.equals(status.index):
        raise ValueError(
            "dataset and status indices must match exactly."
        )
    if not dataset.index.equals(splits.index):
        raise ValueError(
            "dataset and splits indices must match exactly."
        )

    if dataset.index.has_duplicates:
        raise ValueError("dataset index must be unique.")

    if "target" not in dataset.columns:
        raise ValueError("dataset must contain target.")

    if "supervised_eligible" not in status.columns:
        raise ValueError(
            "status must contain supervised_eligible."
        )

    if "period" not in splits.columns:
        raise ValueError("splits must contain period.")

    if status["supervised_eligible"].isna().any():
        raise ValueError(
            "supervised_eligible must not contain missing values."
        )


def build_development_folds(
    dataset: pd.DataFrame,
    status: pd.DataFrame,
    splits: pd.DataFrame,
) -> dict[str, DevelopmentFold]:
    """Materialize the three frozen development folds.

    Returned indices contain supervised-eligible rows only. Protected Final
    Test timestamps are rejected even if upstream split masks were corrupted.
    """

    _validate_aligned(dataset, status, splits)

    raw_folds = development_fold_masks(splits)
    eligible = status["supervised_eligible"].astype(bool)
    final_test = splits["period"].eq(PERIOD_FINAL_TEST)

    result: dict[str, DevelopmentFold] = {}

    for name in DEVELOPMENT_FOLD_NAMES:
        masks = raw_folds[name]

        train_mask = masks["train"] & eligible
        validation_mask = masks["validation"] & eligible

        if (train_mask & final_test).any():
            raise AssertionError(
                f"Final Test entered {name} training rows."
            )
        if (validation_mask & final_test).any():
            raise AssertionError(
                f"Final Test entered {name} validation rows."
            )

        train_index = pd.DatetimeIndex(
            dataset.index[train_mask],
            name=dataset.index.name,
        )
        validation_index = pd.DatetimeIndex(
            dataset.index[validation_mask],
            name=dataset.index.name,
        )

        if len(train_index) and len(validation_index):
            if train_index.max() >= validation_index.min():
                raise AssertionError(
                    f"{name} is not strictly chronological."
                )

        result[name] = DevelopmentFold(
            name=name,
            train_index=train_index,
            validation_index=validation_index,
        )

    return result


def get_development_xy(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
    feature_columns: tuple[str, ...] | list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return X_train, y_train, X_validation, y_validation for one fold.

    Only explicitly requested feature columns are exposed. This prevents audit,
    status, target, or future-derived metadata from entering X accidentally.
    """

    columns = list(feature_columns)

    if not columns:
        raise ValueError("feature_columns must not be empty.")

    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns must be unique.")

    if "target" in columns:
        raise ValueError("target must not be included in feature_columns.")

    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(
            f"dataset is missing requested feature columns: {missing}"
        )

    train = dataset.loc[fold.train_index]
    validation = dataset.loc[fold.validation_index]

    x_train = train.loc[:, columns].copy()
    y_train = train["target"].copy()
    x_validation = validation.loc[:, columns].copy()
    y_validation = validation["target"].copy()

    if y_train.isna().any() or y_validation.isna().any():
        raise AssertionError(
            "DevelopmentFold contains an unknown target."
        )

    return x_train, y_train, x_validation, y_validation
