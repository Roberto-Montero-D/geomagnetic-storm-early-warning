"""Causal time-based rolling features for primary OMNI predictors.

Windows are defined in physical time, never by row count.

For prediction time t:
    cutoff = t - 1h

For a W-hour rolling window, eligible OMNI hourly intervals satisfy:
    cutoff - W < period_end <= cutoff

Only observations already causally available at t enter the window.
Missing timestamps are not replaced and do not cause the window to extend
farther into the past.

This implementation preserves the frozen semantics while using pandas'
vectorized time-based rolling engine.
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
    information_cutoff,
    interval_end_times,
)

ROLLING_WINDOWS_HOURS = (3, 6, 12, 24)
ROLLING_STATISTICS = ("mean", "min", "std")


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


def _prepare_omni(omni: pd.DataFrame) -> pd.DataFrame:
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


def rolling_feature_names(
    windows_hours: tuple[int, ...] = ROLLING_WINDOWS_HOURS,
) -> tuple[str, ...]:
    """Return deterministic rolling-feature column order."""
    return tuple(
        f"{column}_roll_{stat}_{window}h"
        for column in PRIMARY_OMNI_COLUMNS
        for window in windows_hours
        for stat in ROLLING_STATISTICS
    )


def build_rolling_features(
    omni: pd.DataFrame,
    prediction_times: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    windows_hours: tuple[int, ...] = ROLLING_WINDOWS_HOURS,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal rolling mean/min/std features over physical-time windows.

    Statistics use the valid observations physically present inside each
    window. Missing source timestamps and missing values are not imputed.
    A window containing no valid values for a variable yields NaN.

    Pandas sample standard deviation semantics are used (ddof=1), so std is
    NaN when fewer than two valid values exist.
    """
    prediction_index = _validate_prediction_times(prediction_times)
    work = _prepare_omni(omni)

    if not windows_hours:
        raise ValueError("windows_hours must not be empty.")
    if any(
        (not isinstance(window, int))
        or isinstance(window, bool)
        or window <= 0
        for window in windows_hours
    ):
        raise ValueError("windows_hours must contain positive integers.")
    if len(set(windows_hours)) != len(windows_hours):
        raise ValueError("windows_hours must be unique.")

    ends = pd.DatetimeIndex(interval_end_times(work.index))

    source = work.copy()
    source.index = ends
    source.index.name = "information_time"

    cutoffs = pd.DatetimeIndex(
        [information_cutoff(t) for t in prediction_index]
    )

    # Add requested cutoffs to the source timeline. An inserted cutoff row is
    # NaN and contributes no observation; it only lets pandas evaluate the
    # physical-time rolling window at the exact requested instant.
    timeline = source.index.union(cutoffs).sort_values()
    aligned = source.reindex(timeline)

    pieces: dict[str, np.ndarray] = {}

    for window in windows_hours:
        roller = aligned.rolling(
            f"{window}h",
            closed="right",
            min_periods=1,
        )

        statistic_frames = {
            "mean": roller.mean(),
            "min": roller.min(),
            "std": roller.std(ddof=1),
        }

        for column in PRIMARY_OMNI_COLUMNS:
            for stat in ROLLING_STATISTICS:
                feature_name = f"{column}_roll_{stat}_{window}h"

                pieces[feature_name] = (
                    statistic_frames[stat][column]
                    .reindex(cutoffs)
                    .to_numpy(dtype=float)
                )

    output = pd.DataFrame(
        pieces,
        index=prediction_index,
        dtype=float,
    )

    output = output.loc[:, rolling_feature_names(windows_hours)]
    output.index.name = "prediction_time"

    audit = pd.DataFrame(index=prediction_index)
    audit.index.name = "prediction_time"

    # Preserve the same datetime-resolution contract used by the rest of the
    # feature pipeline. Do not force datetime64[ns].
    audit["information_cutoff"] = pd.Series(
        [information_cutoff(t) for t in prediction_index],
        index=prediction_index,
        name="information_cutoff",
    )

    audit["maximum_rolling_information_time"] = pd.NaT

    # Windows are nested. Therefore the maximum provenance across all rolling
    # features is the latest valid source interval end lying inside the largest
    # requested window.
    valid_ends = source.index[
        source.notna().any(axis=1).to_numpy()
    ]

    if len(valid_ends) and len(cutoffs):
        positions = (
            valid_ends.searchsorted(
                cutoffs,
                side="right",
            )
            - 1
        )

        has_candidate = positions >= 0

        # Preserve the datetime unit already used by the source index rather
        # than coercing provenance to nanoseconds.
        source_datetime_dtype = valid_ends.to_numpy().dtype

        latest = np.full(
            len(cutoffs),
            np.array("NaT", dtype=source_datetime_dtype),
            dtype=source_datetime_dtype,
        )

        if has_candidate.any():
            latest[has_candidate] = (
                valid_ends.to_numpy()[
                    positions[has_candidate]
                ]
            )

        lower_bound = (
            cutoffs
            - pd.Timedelta(hours=max(windows_hours))
        ).to_numpy(dtype=source_datetime_dtype)

        in_largest_window = (
            has_candidate
            & (latest > lower_bound)
        )

        latest[~in_largest_window] = np.array(
            "NaT",
            dtype=source_datetime_dtype,
        )

        audit["maximum_rolling_information_time"] = pd.Series(
            latest,
            index=prediction_index,
        )

    violation = (
        audit["maximum_rolling_information_time"].notna()
        & (
            audit["maximum_rolling_information_time"]
            > audit["information_cutoff"]
        )
    )

    if violation.any():
        raise AssertionError(
            "Rolling feature provenance exceeds the information cutoff."
        )

    if return_audit:
        return output, audit

    return output
