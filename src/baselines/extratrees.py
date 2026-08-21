"""B3 ExtraTrees baseline.

Frozen baseline scope:
- raw primary predictors only;
- ExtraTrees classifier;
- no balancing;
- validation probabilities for later operational evaluation.

MASTER_PROTOCOL_v1.3.md does not freeze a numerical baseline ExtraTrees
configuration. Therefore this implementation requires n_estimators and
max_depth explicitly instead of silently borrowing the later Phase 5 search
grid or sklearn defaults.

A fixed random_state is used only for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from src.baselines.framework import DevelopmentFold, get_development_xy
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS


EXTRATREES_FEATURES = tuple(PRIMARY_RAW_FEATURE_COLUMNS)
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class ExtraTreesFoldResult:
    """Fitted B3 model and validation probabilities for one fold."""

    fold_name: str
    model: ExtraTreesClassifier
    validation_probability: pd.Series


def make_extratrees_model(
    *,
    n_estimators: int,
    max_depth: int | None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ExtraTreesClassifier:
    """Construct one explicit unbalanced B3 ExtraTrees model."""

    if not isinstance(n_estimators, int) or isinstance(n_estimators, bool):
        raise TypeError("n_estimators must be an integer.")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")

    if max_depth is not None:
        if not isinstance(max_depth, int) or isinstance(max_depth, bool):
            raise TypeError("max_depth must be an integer or None.")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive or None.")

    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )


def fit_extratrees_fold(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
    *,
    n_estimators: int,
    max_depth: int | None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ExtraTreesFoldResult:
    """Fit B3 on one materialized development fold."""

    x_train, y_train, x_validation, _ = get_development_xy(
        dataset,
        fold,
        EXTRATREES_FEATURES,
    )

    if x_train.isna().any().any():
        raise AssertionError(
            "B3 training data contains missing raw predictors."
        )
    if x_validation.isna().any().any():
        raise AssertionError(
            "B3 validation data contains missing raw predictors."
        )

    classes = np.unique(y_train.to_numpy())
    if len(classes) != 2:
        raise ValueError(
            "B3 training target must contain both classes."
        )

    model = make_extratrees_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(x_train, y_train.astype(int))

    probability = pd.Series(
        model.predict_proba(x_validation)[:, 1],
        index=x_validation.index,
        name="probability",
        dtype=float,
    )

    if not probability.between(0.0, 1.0).all():
        raise AssertionError(
            "B3 validation probabilities must lie in [0, 1]."
        )

    return ExtraTreesFoldResult(
        fold_name=fold.name,
        model=model,
        validation_probability=probability,
    )
