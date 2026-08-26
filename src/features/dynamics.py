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

All required timestamps must exist and all required values must be valid.
Missing timestamps, missing values, or source fill values produce NaN for the
affected dynamic feature. The builder never substitutes a nearby row.

This implementation preserves those semantics using vectorized exact reindexing
and the same centered OLS arithmetic as the original reference implementation.
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


def _ols_slope_four_points(
    oldest: np.ndarray,
    older_2: np.ndarray,
    older_1: np.ndarray,
    latest: np.ndarray,
) -> np.ndarray:
    """Return the frozen four-point OLS slope in units per hour.

    This intentionally mirrors the original implementation's arithmetic:
    center x and y, then compute dot(x_centered, y_centered) / denominator.

    Using a pre-collapsed weight vector is mathematically equivalent but can
    differ by a few floating-point ULPs, which would violate the existing
    regression contract for exact simple-linear cases.
    """
    matrix = np.column_stack(
        [oldest, older_2, older_1, latest]
    ).astype(float, copy=False)

    missing = np.isnan(matrix).any(axis=1)

    x = np.arange(4, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))

    slopes = np.full(
        matrix.shape[0],
        np.nan,
        dtype=float,
    )

    valid = ~missing
    if valid.any():
        valid_matrix = matrix[valid]

        # Reproduce the reference implementation row-wise, but vectorized:
        # y_centered = y - mean(y)
        y_centered = (
            valid_matrix
            - valid_matrix.mean(axis=1, keepdims=True)
        )

        numerators = y_centered @ x_centered

        slopes[valid] = (
            numerators
            / denominator
        )

    return slopes


def build_dynamic_features(
    omni: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal 1h/3h deltas and 3h OLS slopes."""
    prediction_index = _validate_prediction_times(prediction_times)
    source = _prepare_source(omni)

    latest_starts = (
        prediction_index
        - pd.Timedelta(hours=2)
    )

    # Exact source timestamp lookups. Missing timestamps remain NaN.
    lag_frames: dict[int, pd.DataFrame] = {}

    for lag in range(4):
        requested_starts = (
            latest_starts
            - pd.Timedelta(hours=lag)
        )

        frame = source.reindex(
            requested_starts
        )

        frame.index = prediction_index
        lag_frames[lag] = frame

    latest = lag_frames[0]

    output_data: dict[str, np.ndarray] = {}

    for column in PRIMARY_OMNI_COLUMNS:
        output_data[
            f"{column}_delta_1h"
        ] = (
            latest[column]
            - lag_frames[1][column]
        ).to_numpy(dtype=float)

        output_data[
            f"{column}_delta_3h"
        ] = (
            latest[column]
            - lag_frames[3][column]
        ).to_numpy(dtype=float)

        output_data[
            f"{column}_slope_3h"
        ] = _ols_slope_four_points(
            lag_frames[3][column].to_numpy(dtype=float),
            lag_frames[2][column].to_numpy(dtype=float),
            lag_frames[1][column].to_numpy(dtype=float),
            lag_frames[0][column].to_numpy(dtype=float),
        )

    features = pd.DataFrame(
        output_data,
        index=prediction_index,
        dtype=float,
    )

    features = features.loc[
        :,
        DYNAMIC_FEATURE_COLUMNS,
    ]
    features.index.name = "prediction_time"

    audit = pd.DataFrame(
        index=prediction_index
    )
    audit.index.name = "prediction_time"

    audit["information_cutoff"] = pd.Series(
        [
            information_cutoff(t)
            for t in prediction_index
        ],
        index=prediction_index,
        name="information_cutoff",
    )

    audit["dynamics_information_time"] = pd.NaT

    source_present = latest_starts.isin(
        source.index
    )

    if source_present.any():
        provenance = pd.Series(
            pd.NaT,
            index=prediction_index,
            dtype="datetime64[ns]",
        )

        provenance.loc[source_present] = (
            latest_starts[source_present]
            + DEFAULT_INTERVAL_DURATION
        ).to_numpy()

        audit[
            "dynamics_information_time"
        ] = provenance

    violation = (
        audit[
            "dynamics_information_time"
        ].notna()
        & (
            audit[
                "dynamics_information_time"
            ]
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
