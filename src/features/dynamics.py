"""Causal dynamic features for primary OMNI predictors.

Frozen dynamics
---------------
For each primary OMNI variable:

    delta_1h = latest_value - value exactly 1 hour earlier
    delta_3h = latest_value - value exactly 3 hours earlier

The 3-hour slope is the ordinary-least-squares slope (units per hour) across
the four exact hourly samples spanning the same 3 elapsed hours:

    t0, t0-1h, t0-2h, t0-3h

where ``t0`` is the latest causally eligible OMNI interval start.

This makes the slope a trend estimate distinct from the endpoint-only 3-hour
delta.

All required timestamps must exist and all required values must be valid.
Missing timestamps, missing values, or source fill values produce NaN for the
affected dynamic feature. The builder never substitutes a nearby row.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.features.raw import (
    PRIMARY_OMNI_COLUMNS,
    PRIMARY_OMNI_FILL_VALUES,
)
from src.temporal.cutoff import (
    DEFAULT_INTERVAL_DURATION,
    information_cutoff,
)

DYNAMIC_DELTAS_HOURS = (1, 3)
DYNAMIC_SLOPE_HOURS = 3

DYNAMIC_FEATURE_COLUMNS = tuple(
    feature
    for column in PRIMARY_OMNI_COLUMNS
    for feature in (
        f"{column}_delta_1h",
        f"{column}_delta_3h",
        f"{column}_slope_3h",
    )
)


def _validate_prediction_times(
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(prediction_times, name="prediction_time")

    if index.hasnans:
        raise ValueError("prediction_times must not contain NaT.")
    if index.has_duplicates:
        raise ValueError("prediction_times must be unique.")
    if not index.is_monotonic_increasing:
        raise ValueError("prediction_times must be monotonically increasing.")
    if len(index) and (
        (index.minute != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (index.nanosecond != 0).any()
    ):
        raise ValueError("prediction_times must be aligned to whole hours.")

    return index


def _prepare_source(omni: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(omni.index, pd.DatetimeIndex):
        raise TypeError("omni.index must be a pandas DatetimeIndex.")
    if omni.index.hasnans:
        raise ValueError("omni.index must not contain NaT.")
    if omni.index.has_duplicates:
        raise ValueError("omni.index must be unique.")
    if not omni.index.is_monotonic_increasing:
        raise ValueError("omni.index must be monotonically increasing.")

    missing = set(PRIMARY_OMNI_COLUMNS) - set(omni.columns)
    if missing:
        raise KeyError(f"Missing primary OMNI column(s): {sorted(missing)}")

    work = omni.loc[:, PRIMARY_OMNI_COLUMNS].copy()
    for column in PRIMARY_OMNI_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="raise")
        work[column] = work[column].mask(
            work[column] == PRIMARY_OMNI_FILL_VALUES[column]
        )

    return work.astype(float)


def _exact_values(
    series: pd.Series,
    timestamps: pd.DatetimeIndex,
) -> np.ndarray | None:
    if not timestamps.isin(series.index).all():
        return None

    values = series.reindex(timestamps).to_numpy(dtype=float)
    if np.isnan(values).any():
        return None

    return values


def _ols_slope_per_hour(values_oldest_to_latest: np.ndarray) -> float:
    x = np.arange(values_oldest_to_latest.size, dtype=float)
    x_centered = x - x.mean()
    y_centered = values_oldest_to_latest - values_oldest_to_latest.mean()

    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        return np.nan

    return float(np.dot(x_centered, y_centered) / denominator)


def build_dynamic_features(
    omni: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal 1h/3h deltas and 3h OLS slopes."""

    prediction_index = _validate_prediction_times(prediction_times)
    source = _prepare_source(omni)

    features = pd.DataFrame(
        np.nan,
        index=prediction_index,
        columns=DYNAMIC_FEATURE_COLUMNS,
        dtype=float,
    )
    features.index.name = "prediction_time"

    audit = pd.DataFrame(index=prediction_index)
    audit.index.name = "prediction_time"
    audit["information_cutoff"] = pd.DatetimeIndex(
        [information_cutoff(t) for t in prediction_index]
    )
    audit["dynamics_information_time"] = pd.NaT

    for row_i, t in enumerate(prediction_index):
        cutoff = information_cutoff(t)
        latest_start = cutoff - DEFAULT_INTERVAL_DURATION

        if latest_start in source.index:
            audit.iat[
                row_i,
                audit.columns.get_loc("dynamics_information_time"),
            ] = latest_start + DEFAULT_INTERVAL_DURATION

        for column in PRIMARY_OMNI_COLUMNS:
            latest = source[column].get(latest_start, np.nan)

            for lag in DYNAMIC_DELTAS_HOURS:
                older_start = latest_start - pd.Timedelta(hours=lag)
                older = source[column].get(older_start, np.nan)

                name = f"{column}_delta_{lag}h"

                if pd.notna(latest) and pd.notna(older):
                    features.iat[
                        row_i,
                        features.columns.get_loc(name),
                    ] = float(latest - older)

            slope_times = pd.date_range(
                latest_start - pd.Timedelta(hours=DYNAMIC_SLOPE_HOURS),
                latest_start,
                freq="h",
            )
            slope_values = _exact_values(source[column], slope_times)

            if slope_values is not None:
                name = f"{column}_slope_3h"
                features.iat[
                    row_i,
                    features.columns.get_loc(name),
                ] = _ols_slope_per_hour(slope_values)

    violation = (
        audit["dynamics_information_time"].notna()
        & (
            audit["dynamics_information_time"]
            > audit["information_cutoff"]
        )
    )
    if violation.any():
        raise AssertionError(
            "Dynamic-feature provenance exceeds the information cutoff."
        )

    if return_audit:
        return features, audit
    return features
