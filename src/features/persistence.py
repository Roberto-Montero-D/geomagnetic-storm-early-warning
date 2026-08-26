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

This implementation preserves those semantics using vectorized run lengths.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.features.raw import PRIMARY_OMNI_FILL_VALUES
from src.temporal.cutoff import (
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
        raise KeyError(
            f"Missing persistence OMNI column(s): {sorted(missing)}"
        )

    work = omni.loc[:, ["bz_gsm", "speed"]].copy()

    for column in work.columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
        work[column] = work[column].mask(
            work[column] == PRIMARY_OMNI_FILL_VALUES[column]
        )

    return work.astype(float)


def _consecutive_true_run(
    values: pd.Series,
    condition: pd.Series,
) -> pd.Series:
    """Return consecutive satisfied hours with frozen missing-state semantics."""
    satisfied = condition & values.notna()

    # False/missing rows reset the group. Within true groups, cumulative sum is
    # exactly the consecutive duration ending at each timestamp.
    group_id = (~satisfied).cumsum()
    run = satisfied.groupby(group_id).cumsum().astype(float)

    # Latest state unknown -> persistence unknown, not zero.
    run.loc[values.isna()] = np.nan
    return run


def build_persistence_features(
    omni: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build frozen causal persistence-duration features."""
    prediction_index = _validate_prediction_times(prediction_times)
    source = _prepare_source(omni)

    latest_starts = prediction_index - pd.Timedelta(hours=2)

    if source.empty:
        features = pd.DataFrame(
            np.nan,
            index=prediction_index,
            columns=PERSISTENCE_FEATURE_COLUMNS,
            dtype=float,
        )
    else:
        # Insert absent physical hours explicitly as NaN so gaps break a run.
        physical_index = pd.date_range(
            source.index.min(),
            source.index.max(),
            freq="h",
        )
        full = source.reindex(physical_index)

        runs = pd.DataFrame(
            {
                "bz_gsm_persist_lt_m5h": _consecutive_true_run(
                    full["bz_gsm"],
                    full["bz_gsm"] < -5.0,
                ),
                "bz_gsm_persist_lt_m10h": _consecutive_true_run(
                    full["bz_gsm"],
                    full["bz_gsm"] < -10.0,
                ),
                "bz_gsm_persist_lt_m15h": _consecutive_true_run(
                    full["bz_gsm"],
                    full["bz_gsm"] < -15.0,
                ),
                "speed_persist_gt_500h": _consecutive_true_run(
                    full["speed"],
                    full["speed"] > 500.0,
                ),
                "speed_persist_gt_600h": _consecutive_true_run(
                    full["speed"],
                    full["speed"] > 600.0,
                ),
            }
        )

        features = runs.reindex(latest_starts)
        features.index = prediction_index

    features = features.loc[:, PERSISTENCE_FEATURE_COLUMNS].astype(float)
    features.index.name = "prediction_time"

    audit = pd.DataFrame(index=prediction_index)
    audit.index.name = "prediction_time"

    audit["information_cutoff"] = pd.Series(
        [information_cutoff(t) for t in prediction_index],
        index=prediction_index,
        name="information_cutoff",
    )
    audit["persistence_information_time"] = pd.NaT

    source_present = latest_starts.isin(source.index)
    if source_present.any():
        provenance = pd.Series(
            pd.NaT,
            index=prediction_index,
            dtype="datetime64[ns]",
        )
        provenance.loc[source_present] = (
            latest_starts[source_present] + DEFAULT_INTERVAL_DURATION
        ).to_numpy()
        audit["persistence_information_time"] = provenance

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
