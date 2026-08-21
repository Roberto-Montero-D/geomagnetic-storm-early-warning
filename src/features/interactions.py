"""Causal interaction features from the latest eligible raw OMNI state.

Frozen interactions
-------------------
    Bz_neg * V
    Bz_neg * Density
    Pressure * V

where:
    Bz_neg = max(-bz_gsm, 0)

The interaction builder consumes the canonical causal raw feature frame rather
than selecting OMNI observations independently. Missing inputs propagate to
NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INTERACTION_FEATURE_COLUMNS = (
    "bz_neg_x_speed",
    "bz_neg_x_density",
    "flow_pressure_x_speed",
)


def build_interaction_features(
    raw_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build the frozen interaction family from causal raw predictors."""

    required = {
        "bz_gsm",
        "speed",
        "density",
        "flow_pressure",
    }
    missing = required - set(raw_features.columns)
    if missing:
        raise KeyError(
            f"Missing raw feature column(s) for interactions: {sorted(missing)}"
        )

    if not isinstance(raw_features.index, pd.DatetimeIndex):
        raise TypeError(
            "raw_features.index must be a pandas DatetimeIndex."
        )
    if raw_features.index.hasnans:
        raise ValueError("raw_features.index must not contain NaT.")
    if raw_features.index.has_duplicates:
        raise ValueError("raw_features.index must be unique.")
    if not raw_features.index.is_monotonic_increasing:
        raise ValueError(
            "raw_features.index must be monotonically increasing."
        )

    bz = pd.to_numeric(raw_features["bz_gsm"], errors="raise")
    speed = pd.to_numeric(raw_features["speed"], errors="raise")
    density = pd.to_numeric(raw_features["density"], errors="raise")
    pressure = pd.to_numeric(
        raw_features["flow_pressure"],
        errors="raise",
    )

    bz_neg = (-bz).clip(lower=0.0)

    output = pd.DataFrame(
        {
            "bz_neg_x_speed": bz_neg * speed,
            "bz_neg_x_density": bz_neg * density,
            "flow_pressure_x_speed": pressure * speed,
        },
        index=raw_features.index.copy(),
    )
    output = output.loc[:, INTERACTION_FEATURE_COLUMNS]
    output.index.name = raw_features.index.name or "prediction_time"

    return output.astype(float)
