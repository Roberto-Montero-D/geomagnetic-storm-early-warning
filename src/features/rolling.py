"""Causal time-based rolling features for primary OMNI predictors.

Windows are defined in physical time, never by row count.

For prediction time t:
    cutoff = t - 1h

For a W-hour rolling window, eligible OMNI hourly intervals satisfy:
    cutoff - W < period_end <= cutoff

Because OMNI timestamps are interval starts and each interval lasts one hour,
this is equivalent to source starts:
    cutoff - W - 1h < period_start <= cutoff - 1h

Only observations already causally available at t enter the window.
Missing timestamps are not replaced and do not cause the window to extend
farther into the past.
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
    if any((not isinstance(w, int)) or isinstance(w, bool) or w <= 0
           for w in windows_hours):
        raise ValueError("windows_hours must contain positive integers.")
    if len(set(windows_hours)) != len(windows_hours):
        raise ValueError("windows_hours must be unique.")

    ends = interval_end_times(work.index)
    values = work.to_numpy(dtype=float)

    output = pd.DataFrame(
        np.nan,
        index=prediction_index,
        columns=rolling_feature_names(windows_hours),
        dtype=float,
    )
    output.index.name = "prediction_time"

    audit = pd.DataFrame(index=prediction_index)
    audit.index.name = "prediction_time"
    audit["information_cutoff"] = pd.DatetimeIndex(
        [information_cutoff(t) for t in prediction_index]
    )
    audit["maximum_rolling_information_time"] = pd.NaT

    ends_np = ends.to_numpy()

    for row_i, t in enumerate(prediction_index):
        cutoff = information_cutoff(t)
        row_max_info = pd.NaT

        for window in windows_hours:
            lower = cutoff - pd.Timedelta(hours=window)

            # Physical-time interval: (cutoff-W, cutoff].
            mask = (ends > lower) & (ends <= cutoff)
            if not mask.any():
                continue

            window_values = values[mask]
            window_ends = ends[mask]

            for col_i, column in enumerate(PRIMARY_OMNI_COLUMNS):
                series = window_values[:, col_i]
                valid = ~np.isnan(series)
                if not valid.any():
                    continue

                valid_values = series[valid]
                output.iat[
                    row_i,
                    output.columns.get_loc(
                        f"{column}_roll_mean_{window}h"
                    ),
                ] = float(np.mean(valid_values))
                output.iat[
                    row_i,
                    output.columns.get_loc(
                        f"{column}_roll_min_{window}h"
                    ),
                ] = float(np.min(valid_values))

                if valid_values.size >= 2:
                    output.iat[
                        row_i,
                        output.columns.get_loc(
                            f"{column}_roll_std_{window}h"
                        ),
                    ] = float(np.std(valid_values, ddof=1))

                latest_valid_end = window_ends[valid][-1]
                if pd.isna(row_max_info) or latest_valid_end > row_max_info:
                    row_max_info = latest_valid_end

        audit.iat[
            row_i,
            audit.columns.get_loc("maximum_rolling_information_time"),
        ] = row_max_info

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
