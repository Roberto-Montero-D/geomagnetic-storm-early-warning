"""Causal construction of the primary raw feature frame.

The builder in this module is deliberately conservative.

For prediction time t, an hourly OMNI measurement beginning at s is usable
only when:

    s + 1h <= t - 1h

For the latest raw OMNI state this means the expected source row begins at
t - 2h.  If that exact source timestamp is absent, the OMNI raw features for
that prediction time remain missing; the builder does not fall back to an
older row and imply continuity.

Predictor-side Kp is constructed only through the canonical Kp helpers:

    kp_lag_h(t) = kp_asof(t - h)

No target or retrospective event information enters this module.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.data.kp import DEFAULT_KP_LAGS_HOURS, build_kp_lag_features
from src.temporal.cutoff import (
    DEFAULT_INFORMATION_DELAY,
    DEFAULT_INTERVAL_DURATION,
    information_cutoff,
    interval_end_times,
)


PRIMARY_OMNI_COLUMNS = (
    "bz_gsm",
    "bt",
    "speed",
    "density",
    "flow_pressure",
)

PRIMARY_RAW_FEATURE_COLUMNS = (
    *PRIMARY_OMNI_COLUMNS,
    "kp_lag_1h",
    "kp_lag_3h",
    "kp_lag_6h",
    "kp_lag_12h",
    "kp_lag_24h",
)

# OMNIWeb fill values for the primary raw solar-wind variables used by the
# project.  Replacement happens at the feature boundary rather than in the
# raw loader so the source file remains auditable.
PRIMARY_OMNI_FILL_VALUES = {
    "bz_gsm": 999.9,
    "bt": 999.9,
    "speed": 9999.0,
    "density": 999.9,
    "flow_pressure": 99.99,
}


def _validate_prediction_times(
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Normalize and validate prediction timestamps."""

    index = pd.DatetimeIndex(prediction_times, name="prediction_time")

    if index.hasnans:
        raise ValueError("prediction_times must not contain NaT.")

    if index.has_duplicates:
        raise ValueError("prediction_times must be unique.")

    if not index.is_monotonic_increasing:
        raise ValueError(
            "prediction_times must be monotonically increasing."
        )

    if len(index) and (
        (index.minute != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (index.nanosecond != 0).any()
    ):
        raise ValueError(
            "prediction_times must be aligned to whole hours."
        )

    return index


def _validate_omni_frame(omni: pd.DataFrame) -> None:
    """Validate the OMNI input required by the raw feature builder."""

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
        raise KeyError(
            f"Missing primary OMNI column(s): {sorted(missing)}"
        )


def _normalize_primary_omni_values(omni: pd.DataFrame) -> pd.DataFrame:
    """Return primary OMNI variables with source fill values mapped to NaN."""

    work = omni.loc[:, PRIMARY_OMNI_COLUMNS].copy()

    for column in PRIMARY_OMNI_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="raise")
        work[column] = work[column].mask(
            work[column] == PRIMARY_OMNI_FILL_VALUES[column]
        )

    return work.astype(float)


def _latest_omni_features(
    omni: pd.DataFrame,
    prediction_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select the exact latest causally eligible hourly OMNI interval.

    For the canonical one-hour interval and one-hour information delay,
    prediction at t uses the source interval beginning at t - 2h.

    Missing source timestamps remain missing rather than causing fallback to
    an older measurement.
    """

    normalized = _normalize_primary_omni_values(omni)

    expected_starts = (
        prediction_index
        - DEFAULT_INFORMATION_DELAY
        - DEFAULT_INTERVAL_DURATION
    )

    selected = normalized.reindex(expected_starts)
    selected.index = prediction_index

    source_present = pd.Series(
        expected_starts.isin(omni.index),
        index=prediction_index,
        dtype=bool,
    )

    information_times = pd.Series(
        pd.NaT,
        index=prediction_index,
        dtype="datetime64[ns]",
        name="omni_information_time",
    )

    if source_present.any():
        present_starts = pd.DatetimeIndex(
            expected_starts[source_present.to_numpy()]
        )
        present_ends = interval_end_times(present_starts)
        information_times.loc[source_present] = present_ends.to_numpy()

    return selected, information_times


def _kp_information_times(
    intervals: pd.DataFrame,
    prediction_index: pd.DatetimeIndex,
    lags_hours: tuple[int, ...],
) -> pd.DataFrame:
    """Return interval-end provenance for each causal Kp lag."""

    required = {"interval_end", "kp"}
    missing = required - set(intervals.columns)
    if missing:
        raise KeyError(
            f"Missing interval column(s): {sorted(missing)}"
        )

    audit = pd.DataFrame(index=prediction_index)

    if intervals.empty:
        for lag in lags_hours:
            audit[f"kp_lag_{lag}h_information_time"] = pd.NaT
        return audit

    ordered = intervals.sort_values("interval_end").reset_index(drop=True)
    ends = pd.DatetimeIndex(ordered["interval_end"])

    if ends.hasnans:
        raise ValueError("Kp interval_end must not contain NaT.")
    if ends.has_duplicates:
        raise ValueError("Kp interval_end must be unique.")
    if not ends.is_monotonic_increasing:
        raise ValueError(
            "Kp interval_end must be monotonically increasing."
        )

    for lag in lags_hours:
        query = prediction_index - pd.Timedelta(hours=lag)
        positions = ends.searchsorted(query, side="right") - 1

        provenance = np.full(
            len(prediction_index),
            np.datetime64("NaT", "ns"),
            dtype="datetime64[ns]",
        )
        valid = positions >= 0
        provenance[valid] = ends.to_numpy()[positions[valid]]

        audit[f"kp_lag_{lag}h_information_time"] = provenance

    return audit


def build_raw_features(
    omni: pd.DataFrame,
    kp_intervals: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build the frozen primary raw causal feature family.

    Parameters
    ----------
    omni
        Raw/loaded OMNI frame indexed by interval-start timestamp.
    kp_intervals
        Canonical Kp intervals produced by ``build_kp_intervals``.
    prediction_times
        Whole-hour operational prediction timestamps.
    return_audit
        If True, also return per-source information-time provenance.

    Returns
    -------
    pandas.DataFrame
        Raw causal predictors indexed by ``prediction_time``.

    or, when ``return_audit=True``:

    (features, audit)

    Notes
    -----
    Missing source values and missing source timestamps are preserved as
    missing features.  This function performs no imputation and drops no
    prediction rows.
    """

    prediction_index = _validate_prediction_times(prediction_times)
    _validate_omni_frame(omni)

    omni_features, omni_information_time = _latest_omni_features(
        omni,
        prediction_index,
    )

    kp_features = build_kp_lag_features(
        kp_intervals,
        prediction_index,
        lags_hours=DEFAULT_KP_LAGS_HOURS,
    )
    kp_features.index = prediction_index

    features = pd.concat(
        [omni_features, kp_features],
        axis=1,
    )
    features = features.loc[:, PRIMARY_RAW_FEATURE_COLUMNS]
    features.index.name = "prediction_time"

    if not return_audit:
        return features

    audit = _kp_information_times(
        kp_intervals,
        prediction_index,
        DEFAULT_KP_LAGS_HOURS,
    )
    audit.insert(
        0,
        "omni_information_time",
        omni_information_time,
    )
    audit.index.name = "prediction_time"

    # Global row-level provenance required by the Data Contract.
    audit["maximum_feature_information_time"] = audit.max(axis=1)

    cutoffs = pd.Series(
        [
            information_cutoff(t)
            for t in prediction_index
        ],
        index=prediction_index,
        name="information_cutoff",
    )
    audit["information_cutoff"] = cutoffs

    violation = (
        audit["maximum_feature_information_time"].notna()
        & (
            audit["maximum_feature_information_time"]
            > audit["information_cutoff"]
        )
    )
    if violation.any():
        raise AssertionError(
            "Causal feature provenance exceeds the information cutoff."
        )

    return features, audit
