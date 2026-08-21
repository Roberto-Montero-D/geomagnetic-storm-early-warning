"""B0 Persistence baseline.

Frozen rule:
    predict storm at prediction time t iff Kp(t - 1h) >= T

The canonical predictor-side Kp feature is ``kp_lag_1h``. Its construction is
owned by Phase 0's raw-feature/Kp helpers; this baseline does not reconstruct
Kp timing from source intervals and never reads the retrospective target.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_STORM_THRESHOLD = 5.0
PERSISTENCE_FEATURE = "kp_lag_1h"


def predict_persistence(
    features: pd.DataFrame,
    *,
    threshold: float = DEFAULT_STORM_THRESHOLD,
) -> pd.Series:
    """Return deterministic B0 binary predictions.

    Missing ``kp_lag_1h`` remains missing in the prediction output rather than
    being silently interpreted as a negative prediction.
    """

    if PERSISTENCE_FEATURE not in features.columns:
        raise ValueError(
            f"features must contain {PERSISTENCE_FEATURE!r}."
        )

    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    kp = pd.to_numeric(
        features[PERSISTENCE_FEATURE],
        errors="raise",
    )

    prediction = pd.Series(
        pd.NA,
        index=features.index,
        dtype="Int8",
        name="prediction",
    )

    known = kp.notna()
    prediction.loc[known] = (
        kp.loc[known] >= float(threshold)
    ).astype("int8")

    return prediction


def predict_persistence_for_index(
    dataset: pd.DataFrame,
    prediction_index: pd.DatetimeIndex,
    *,
    threshold: float = DEFAULT_STORM_THRESHOLD,
) -> pd.Series:
    """Apply B0 to an explicitly materialized development index."""

    missing = prediction_index.difference(dataset.index)
    if len(missing):
        raise ValueError(
            "prediction_index contains timestamps absent from dataset."
        )

    features = dataset.loc[
        prediction_index,
        [PERSISTENCE_FEATURE],
    ]
    return predict_persistence(
        features,
        threshold=threshold,
    )
