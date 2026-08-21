"""B1 Physical baseline.

Frozen protocol structure:
    alert/predict when Bz < -X AND V > Y

MASTER_PROTOCOL_v1.3.md does not freeze numerical values for X or Y.
Therefore this implementation deliberately requires both thresholds explicitly
and does not invent, tune, search, or select them.

The baseline uses the canonical raw Phase 0 features:
    bz_gsm
    speed

No target, future truth, audit metadata, or Final Test information is read.
"""

from __future__ import annotations

import math

import pandas as pd


PHYSICAL_FEATURES = ("bz_gsm", "speed")


def _validate_thresholds(
    bz_magnitude_nt: float,
    speed_threshold_km_s: float,
) -> tuple[float, float]:
    try:
        bz_threshold = float(bz_magnitude_nt)
        speed_threshold = float(speed_threshold_km_s)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Physical thresholds must be numeric."
        ) from exc

    if not math.isfinite(bz_threshold):
        raise ValueError("bz_magnitude_nt must be finite.")
    if not math.isfinite(speed_threshold):
        raise ValueError("speed_threshold_km_s must be finite.")
    if bz_threshold <= 0:
        raise ValueError("bz_magnitude_nt must be positive.")
    if speed_threshold <= 0:
        raise ValueError("speed_threshold_km_s must be positive.")

    return bz_threshold, speed_threshold


def predict_physical(
    features: pd.DataFrame,
    *,
    bz_magnitude_nt: float,
    speed_threshold_km_s: float,
) -> pd.Series:
    """Return deterministic B1 binary predictions.

    Rule:
        1 iff bz_gsm < -bz_magnitude_nt
              AND speed > speed_threshold_km_s

    Both inequalities are strict because the frozen protocol states:
        Bz < -X AND V > Y

    If either required feature is missing, the prediction remains missing
    rather than being silently converted to a negative.
    """

    missing_columns = [
        column
        for column in PHYSICAL_FEATURES
        if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(
            "features is missing required physical columns: "
            f"{missing_columns}"
        )

    bz_threshold, speed_threshold = _validate_thresholds(
        bz_magnitude_nt,
        speed_threshold_km_s,
    )

    bz = pd.to_numeric(features["bz_gsm"], errors="raise")
    speed = pd.to_numeric(features["speed"], errors="raise")

    prediction = pd.Series(
        pd.NA,
        index=features.index,
        dtype="Int8",
        name="prediction",
    )

    known = bz.notna() & speed.notna()
    prediction.loc[known] = (
        (bz.loc[known] < -bz_threshold)
        & (speed.loc[known] > speed_threshold)
    ).astype("int8")

    return prediction


def predict_physical_for_index(
    dataset: pd.DataFrame,
    prediction_index: pd.DatetimeIndex,
    *,
    bz_magnitude_nt: float,
    speed_threshold_km_s: float,
) -> pd.Series:
    """Apply B1 to an explicitly materialized development index."""

    missing = prediction_index.difference(dataset.index)
    if len(missing):
        raise ValueError(
            "prediction_index contains timestamps absent from dataset."
        )

    features = dataset.loc[
        prediction_index,
        list(PHYSICAL_FEATURES),
    ]

    return predict_physical(
        features,
        bz_magnitude_nt=bz_magnitude_nt,
        speed_threshold_km_s=speed_threshold_km_s,
    )
