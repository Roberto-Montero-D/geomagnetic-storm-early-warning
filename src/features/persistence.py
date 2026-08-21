"""Causal persistence-duration features for primary OMNI variables.

Persistence is evaluated at prediction time ``t`` using only information
available by:

    information_cutoff = t - 1h

Because OMNI timestamps mark hourly interval starts, the latest eligible
source interval begins at ``t - 2h``.

Frozen persistence conditions
-----------------------------
Bz:
    bz_gsm < -5 nT
    bz_gsm < -10 nT
    bz_gsm < -15 nT

Solar-wind speed:
    speed > 500 km/s
    speed > 600 km/s

The feature value is the number of consecutive valid hourly intervals,
ending at the latest eligible interval, that satisfy the condition.

Missing source timestamps or missing values break an active persistence run.
If the latest eligible source timestamp/value itself is unavailable, the
persistence feature is NaN rather than zero because the current condition is
unknown.

No imputation or fallback to older timestamps is performed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.features.raw import PRIMARY_OMNI_FILL_VALUES
from src.temporal.cutoff import (
    DEFAULT_INFORMATION_DELAY,
    DEFAULT_INTERVAL_DURATION,
    information_cutoff,
)

BZ_PERSISTENCE_THRESHOLDS = (-5.0, -10.0, -15.0)
SPEED_PERSISTENCE_THRESHOLDS = (500.0, 600.0)

PERSISTENCE_FEATURE_COLUMNS = (
    "bz_gsm_persist_lt_m5h",
    "bz_gsm_persist_lt_m10h",
    "bz_gsm_persist_lt_m15h",
    "speed_persist_gt_500h",
    "speed_persist_gt_600h",
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

    required = {"bz_gsm", "speed"}
    missing = required - set(omni.columns)
    if missing:
        raise KeyError(f"Missing persistence OMNI column(s): {sorted(missing)}")

    work = omni.loc[:, ["bz_gsm", "speed"]].copy()

    for column in work.columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
        work[column] = work[column].mask(
            work[column] == PRIMARY_OMNI_FILL_VALUES[column]
        )

    return work.astype(float)


def _consecutive_duration(
    series: pd.Series,
    latest_start: pd.Timestamp,
    predicate,
) -> float:
    """Return consecutive satisfied hours ending at ``latest_start``.

    Missing latest state -> NaN.
    Latest valid but condition false -> 0.
    Missing older timestamp/value breaks the run.
    """

    if latest_start not in series.index:
        return np.nan

    latest_value = series.loc[latest_start]
    if pd.isna(latest_value):
        return np.nan

    if not predicate(float(latest_value)):
        return 0.0

    duration = 0
    cursor = latest_start

    while True:
        if cursor not in series.index:
            break

        value = series.loc[cursor]
        if pd.isna(value):
            break

        if not predicate(float(value)):
            break

        duration += 1
        cursor -= pd.Timedelta(hours=1)

    return float(duration)


def build_persistence_features(
    omni: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build frozen causal persistence-duration features."""

    prediction_index = _validate_prediction_times(prediction_times)
    source = _prepare_source(omni)

    features = pd.DataFrame(
        np.nan,
        index=prediction_index,
        columns=PERSISTENCE_FEATURE_COLUMNS,
        dtype=float,
    )
    features.index.name = "prediction_time"

    audit = pd.DataFrame(index=prediction_index)
    audit.index.name = "prediction_time"
    audit["information_cutoff"] = pd.DatetimeIndex(
        [information_cutoff(t) for t in prediction_index]
    )
    audit["persistence_information_time"] = pd.NaT

    for row_i, t in enumerate(prediction_index):
        cutoff = information_cutoff(t)
        latest_start = (
            cutoff - DEFAULT_INTERVAL_DURATION
        )

        latest_present = latest_start in source.index

        if latest_present:
            audit.iat[
                row_i,
                audit.columns.get_loc("persistence_information_time"),
            ] = latest_start + DEFAULT_INTERVAL_DURATION

        predicates = (
            lambda x: x < -5.0,
            lambda x: x < -10.0,
            lambda x: x < -15.0,
            lambda x: x > 500.0,
            lambda x: x > 600.0,
        )
        source_columns = (
            "bz_gsm",
            "bz_gsm",
            "bz_gsm",
            "speed",
            "speed",
        )

        for col_i, (source_column, predicate) in enumerate(
            zip(source_columns, predicates, strict=True)
        ):
            features.iat[row_i, col_i] = _consecutive_duration(
                source[source_column],
                latest_start,
                predicate,
            )

    violation = (
        audit["persistence_information_time"].notna()
        & (
            audit["persistence_information_time"]
            > audit["information_cutoff"]
        )
    )
    if violation.any():
        raise AssertionError(
            "Persistence provenance exceeds the information cutoff."
        )

    if return_audit:
        return features, audit
    return features
