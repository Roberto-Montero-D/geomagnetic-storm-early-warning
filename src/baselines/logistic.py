"""B2 Logistic Regression baseline.

Frozen baseline scope:
- raw primary predictors only;
- unbalanced LogisticRegression;
- train-only scaling;
- validation probabilities for later operational threshold evaluation.

No threshold is selected here and Final Test is never materialized here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.framework import DevelopmentFold, get_development_xy
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS


LOGISTIC_FEATURES = tuple(PRIMARY_RAW_FEATURE_COLUMNS)


@dataclass(frozen=True)
class LogisticFoldResult:
    """Fitted B2 pipeline and validation probabilities for one fold."""

    fold_name: str
    model: Pipeline
    validation_probability: pd.Series


def make_logistic_pipeline() -> Pipeline:
    """Construct the fixed unbalanced B2 pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=None,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def fit_logistic_fold(
    dataset: pd.DataFrame,
    fold: DevelopmentFold,
) -> LogisticFoldResult:
    """Fit B2 on one materialized development fold.

    The scaler and classifier are fitted only on fold.train_index.
    Validation rows are transformed/predicted only after fitting.
    """

    x_train, y_train, x_validation, _ = get_development_xy(
        dataset,
        fold,
        LOGISTIC_FEATURES,
    )

    if x_train.isna().any().any():
        raise AssertionError(
            "B2 training data contains missing raw predictors."
        )
    if x_validation.isna().any().any():
        raise AssertionError(
            "B2 validation data contains missing raw predictors."
        )

    classes = np.unique(y_train.to_numpy())
    if len(classes) != 2:
        raise ValueError(
            "B2 training target must contain both classes."
        )

    model = make_logistic_pipeline()
    model.fit(x_train, y_train.astype(int))

    probability = pd.Series(
        model.predict_proba(x_validation)[:, 1],
        index=x_validation.index,
        name="probability",
        dtype=float,
    )

    if not probability.between(0.0, 1.0).all():
        raise AssertionError(
            "B2 validation probabilities must lie in [0, 1]."
        )

    return LogisticFoldResult(
        fold_name=fold.name,
        model=model,
        validation_probability=probability,
    )
