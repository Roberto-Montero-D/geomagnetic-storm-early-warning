"""Reference-equivalence tests for optimized Kp, event, and target engines.

These tests deliberately retain simple loop-based reference implementations
matching the pre-optimization semantics. Their purpose is to prove that the
vectorized production implementations are computational refactors only.
"""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.data.kp import build_kp_intervals
from src.definitions.events import _expand_to_hourly
from src.targets.event_window import (
    _expand_retrospective_kp_hourly,
    build_event_window_target,
)


def _reference_build_kp_intervals(df, kp_column="kp_raw"):
    """Pre-optimization loop semantics for canonical Kp construction."""
    numeric = pd.to_numeric(df[kp_column], errors="raise").astype(float)
    numeric = numeric.mask(numeric == 99)

    work = pd.DataFrame({"kp": numeric}, index=df.index)
    work["interval_start"] = work.index.floor("3h")
    records = []

    for interval_start, group in work.groupby("interval_start", sort=True):
        expected = pd.date_range(interval_start, periods=3, freq="h")
        if len(group) != 3 or not group.index.equals(expected):
            raise ValueError("incomplete interval")

        values = group["kp"]
        if values.isna().all():
            kp_value = np.nan
        elif values.isna().any():
            raise ValueError("partially missing interval")
        else:
            unique = values.unique()
            if len(unique) != 1:
                raise ValueError("inconsistent interval")
            kp_value = float(unique[0]) / 10.0

        # The reference receives raw Kp*10 values. All-missing remains NaN.
        records.append(
            {
                "interval_start": interval_start,
                "interval_end": interval_start + pd.Timedelta(hours=3),
                "kp": kp_value,
            }
        )

    return pd.DataFrame.from_records(records)


def _reference_expand_events(intervals):
    """Pre-optimization hourly event expansion."""
    if intervals.empty:
        return pd.Series(dtype=float)

    index = pd.date_range(
        intervals["interval_start"].iloc[0],
        intervals["interval_end"].iloc[-1],
        freq="h",
        inclusive="left",
    )
    hourly = pd.Series(np.nan, index=index, dtype=float)

    for row in intervals.itertuples(index=False):
        hours = pd.date_range(
            row.interval_start,
            row.interval_end,
            freq="h",
            inclusive="left",
        )
        hourly.loc[hours] = row.kp

    return hourly


def _reference_expand_target(intervals):
    """Pre-optimization retrospective target expansion."""
    hourly = _reference_expand_events(intervals)
    hourly.index.name = "timestamp"
    hourly.name = "kp"
    return hourly


def _reference_event_window_target(
    intervals,
    prediction_times,
    *,
    threshold=5.0,
    horizon_hours=6,
):
    """Pre-optimization row-wise Event-in-Window target semantics."""
    prediction_index = pd.DatetimeIndex(
        prediction_times,
        name="prediction_time",
    )
    hourly = _reference_expand_target(intervals)

    target = pd.Series(
        np.nan,
        index=prediction_index,
        dtype=float,
        name="target",
    )
    audit = pd.DataFrame(index=prediction_index)
    audit.index.name = "prediction_time"
    audit["future_window_start"] = prediction_index + pd.Timedelta(hours=1)
    audit["future_window_end"] = (
        prediction_index + pd.Timedelta(hours=horizon_hours)
    )
    audit["expected_future_hours"] = horizon_hours
    audit["observed_future_hours"] = 0
    audit["missing_future_hours"] = horizon_hours
    audit["positive_future_hours"] = 0
    audit["target_status"] = "unknown"

    for row_i, t in enumerate(prediction_index):
        future_times = pd.date_range(
            t + pd.Timedelta(hours=1),
            periods=horizon_hours,
            freq="h",
        )
        values = hourly.reindex(future_times)
        observed = values.notna()
        positive = observed & (values >= threshold)

        observed_count = int(observed.sum())
        positive_count = int(positive.sum())
        missing_count = horizon_hours - observed_count

        audit.iloc[row_i, audit.columns.get_loc("observed_future_hours")] = (
            observed_count
        )
        audit.iloc[row_i, audit.columns.get_loc("missing_future_hours")] = (
            missing_count
        )
        audit.iloc[row_i, audit.columns.get_loc("positive_future_hours")] = (
            positive_count
        )

        if positive_count > 0:
            target.iloc[row_i] = 1.0
            audit.iloc[row_i, audit.columns.get_loc("target_status")] = (
                "positive"
            )
        elif missing_count == 0:
            target.iloc[row_i] = 0.0
            audit.iloc[row_i, audit.columns.get_loc("target_status")] = (
                "negative"
            )

    return target, audit


def _raw_kp_fixture():
    index = pd.date_range("2020-01-01", periods=30, freq="h")
    codes = np.repeat(
        np.array([10, 20, 30, 50, 60, 40, 99, 20, 70, 10]),
        3,
    )
    return pd.DataFrame({"kp_raw": codes}, index=index)


def _canonical_gap_fixture():
    starts = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 03:00",
            # Deliberate missing 06:00-09:00 interval.
            "2020-01-01 09:00",
            "2020-01-01 12:00",
            "2020-01-01 15:00",
            "2020-01-01 18:00",
        ]
    )
    return pd.DataFrame(
        {
            "interval_start": starts,
            "interval_end": starts + pd.Timedelta(hours=3),
            "kp": [2.0, 5.0, np.nan, 6.0, 3.0, 2.0],
        }
    )


def test_vectorized_kp_interval_builder_matches_reference():
    raw = _raw_kp_fixture()

    expected = _reference_build_kp_intervals(raw)
    actual = build_kp_intervals(raw)

    assert_frame_equal(actual, expected)


def test_vectorized_event_hourly_expansion_matches_reference():
    intervals = _canonical_gap_fixture()

    expected = _reference_expand_events(intervals)
    actual_states = _expand_to_hourly(intervals)
    actual = pd.Series(
        [state.kp for state in actual_states],
        index=pd.DatetimeIndex([state.timestamp for state in actual_states]),
        dtype=float,
    )

    assert_series_equal(actual, expected)


def test_vectorized_target_hourly_expansion_matches_reference():
    intervals = _canonical_gap_fixture()

    expected = _reference_expand_target(intervals)
    actual = _expand_retrospective_kp_hourly(intervals)

    assert_series_equal(actual, expected)


def test_vectorized_event_window_target_matches_reference():
    intervals = _canonical_gap_fixture()
    prediction_times = pd.date_range(
        "2020-01-01 00:00",
        "2020-01-01 19:00",
        freq="h",
    )

    expected_target, expected_audit = _reference_event_window_target(
        intervals,
        prediction_times,
    )
    actual_target, actual_audit = build_event_window_target(
        intervals,
        prediction_times,
        return_audit=True,
    )

    assert_series_equal(actual_target, expected_target)
    assert_frame_equal(actual_audit, expected_audit)
