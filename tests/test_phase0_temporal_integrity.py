"""Project-wide Phase 0 leakage and temporal-integrity tests.

These tests intentionally exercise the canonical feature and target pipelines
together. They verify separation of predictor information from retrospective
future ground truth, including around temporal boundaries.

Phase 0.9 does not introduce new methodological rules. It verifies that the
rules frozen in Phases 0.1-0.8 remain true when the components are composed.
"""

import numpy as np
import pandas as pd

from src.data.kp import build_kp_intervals
from src.features.integrated import (
    PRIMARY_FEATURE_COLUMNS,
    build_primary_feature_frame,
)
from src.targets.event_window import build_event_window_target


def _omni(
    *,
    start="2020-01-01 00:00",
    periods=96,
):
    index = pd.date_range(
        start,
        periods=periods,
        freq="h",
    )
    x = np.arange(periods, dtype=float)

    # Kp is deliberately simple and valid: each canonical 3h interval
    # contains three identical raw Kp*10 codes.
    kp_codes = np.resize(
        np.repeat(
            [10, 20, 30, 20, 10, 10, 20, 30],
            3,
        ),
        periods,
    )

    frame = pd.DataFrame(
        {
            "bz_gsm": -2.0 - (x % 10),
            "bt": 5.0 + x * 0.05,
            "speed": 400.0 + x * 2.0,
            "density": 5.0 + x * 0.02,
            "flow_pressure": 1.0 + x * 0.01,
            "kp_raw": kp_codes,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def _constant_kp_intervals(
    *,
    start="2020-01-01 00:00",
    periods=16,
    value=1.0,
):
    starts = pd.date_range(
        start,
        periods=periods,
        freq="3h",
    )
    return pd.DataFrame(
        {
            "interval_start": starts,
            "interval_end": starts + pd.Timedelta(hours=3),
            "kp": float(value),
        }
    )


def test_integrated_manifest_remains_frozen_at_93_features():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    x = build_primary_feature_frame(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-02 12:00"]),
    )

    assert tuple(x.columns) == PRIMARY_FEATURE_COLUMNS
    assert x.shape == (1, 93)
    assert not x.columns.has_duplicates


def test_every_integrated_feature_row_respects_t_minus_1h_cutoff():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    prediction_times = pd.date_range(
        "2020-01-01 12:00",
        "2020-01-03 12:00",
        freq="h",
    )

    _, audit = build_primary_feature_frame(
        omni,
        intervals,
        prediction_times,
        return_audit=True,
    )

    expected_cutoff = (
        prediction_times
        - pd.Timedelta(hours=1)
    )

    pd.testing.assert_index_equal(
        pd.DatetimeIndex(
            audit["information_cutoff"],
        ),
        pd.DatetimeIndex(expected_cutoff),
        check_names=False,
    )

    valid = audit[
        "maximum_feature_information_time"
    ].notna()

    assert (
        audit.loc[
            valid,
            "maximum_feature_information_time",
        ]
        <= audit.loc[
            valid,
            "information_cutoff",
        ]
    ).all()


def test_future_omni_cannot_change_x_or_y_at_prediction_time():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])
    t = pd.Timestamp("2020-01-02 00:00")
    times = pd.DatetimeIndex([t])

    x_before = build_primary_feature_frame(
        omni,
        intervals,
        times,
    )
    y_before = build_event_window_target(
        intervals,
        times,
    )

    mutated = omni.copy()

    # At t=00:00 the predictor information cutoff is t-1h.
    # OMNI rows starting at t-1h or later are not eligible because their
    # hourly intervals end after the cutoff.
    mutated.loc[
        mutated.index >= t - pd.Timedelta(hours=1),
        ["bz_gsm", "bt", "speed", "density", "flow_pressure"],
    ] = 123456.0

    x_after = build_primary_feature_frame(
        mutated,
        intervals,
        times,
    )
    y_after = build_event_window_target(
        intervals,
        times,
    )

    pd.testing.assert_frame_equal(
        x_before,
        x_after,
    )
    pd.testing.assert_series_equal(
        y_before,
        y_after,
    )


def test_future_kp_inside_target_window_can_change_y_but_not_x():
    omni = _omni()
    intervals = _constant_kp_intervals(
        periods=16,
        value=1.0,
    )

    t = pd.Timestamp("2020-01-01 12:00")
    times = pd.DatetimeIndex([t])

    x_before = build_primary_feature_frame(
        omni,
        intervals,
        times,
    )
    y_before = build_event_window_target(
        intervals,
        times,
    )

    assert y_before.iloc[0] == 0.0

    mutated = intervals.copy()

    # [15:00,18:00) contributes future ground-truth states 15,16,17,
    # all inside the target window (12:00,18:00].
    mask = (
        mutated["interval_start"]
        == pd.Timestamp("2020-01-01 15:00")
    )
    mutated.loc[mask, "kp"] = 6.0

    x_after = build_primary_feature_frame(
        omni,
        mutated,
        times,
    )
    y_after = build_event_window_target(
        mutated,
        times,
    )

    # Future Kp is allowed to define y, never X.
    pd.testing.assert_frame_equal(
        x_before,
        x_after,
    )
    assert y_after.iloc[0] == 1.0


def test_predictor_eligible_kp_can_change_x_without_changing_y():
    omni = _omni()
    intervals = _constant_kp_intervals(
        periods=16,
        value=1.0,
    )

    t = pd.Timestamp("2020-01-01 12:00")
    times = pd.DatetimeIndex([t])

    x_before = build_primary_feature_frame(
        omni,
        intervals,
        times,
    )
    y_before = build_event_window_target(
        intervals,
        times,
    )

    mutated = intervals.copy()

    # kp_lag_1h(t) queries 11:00, so the most recent completed interval
    # is [06:00,09:00). Mutating it is predictor-eligible.
    mask = (
        mutated["interval_start"]
        == pd.Timestamp("2020-01-01 06:00")
    )
    mutated.loc[mask, "kp"] = 4.0

    x_after = build_primary_feature_frame(
        omni,
        mutated,
        times,
    )
    y_after = build_event_window_target(
        mutated,
        times,
    )

    assert (
        x_before.iloc[0]["kp_lag_1h"]
        != x_after.iloc[0]["kp_lag_1h"]
    )

    # Past predictor Kp must not contaminate future target truth.
    pd.testing.assert_series_equal(
        y_before,
        y_after,
    )


def test_kp_interval_not_completed_by_query_cannot_enter_kp_lag():
    omni = _omni()
    intervals = _constant_kp_intervals(
        periods=16,
        value=1.0,
    )

    t = pd.Timestamp("2020-01-01 12:00")
    times = pd.DatetimeIndex([t])

    x_before = build_primary_feature_frame(
        omni,
        intervals,
        times,
    )

    mutated = intervals.copy()

    # For kp_lag_1h(t), query time is 11:00.
    # Interval [09:00,12:00) is not completed by 11:00 and must not enter.
    mask = (
        mutated["interval_start"]
        == pd.Timestamp("2020-01-01 09:00")
    )
    mutated.loc[mask, "kp"] = 9.0

    x_after = build_primary_feature_frame(
        omni,
        mutated,
        times,
    )

    pd.testing.assert_frame_equal(
        x_before,
        x_after,
    )


def test_target_excludes_t_and_includes_t_plus_h_in_composed_pipeline():
    omni = _omni()
    intervals = _constant_kp_intervals(
        periods=16,
        value=1.0,
    )

    t = pd.Timestamp("2020-01-01 12:00")
    times = pd.DatetimeIndex([t])

    # Make current time storm-level through interval [12:00,15:00).
    # Because 13:00 and 14:00 are also future states, this interval alone
    # cannot isolate the left-boundary test at an arbitrary t. Instead use a
    # prediction at the final hour of a storm interval below.
    mutated = intervals.copy()
    mutated.loc[
        mutated["interval_start"]
        == pd.Timestamp("2020-01-01 09:00"),
        "kp",
    ] = 6.0

    # At t=11:00, Kp(t) is storm-level from [09:00,12:00), but the target
    # window begins at 12:00, so current-time storm state is excluded.
    y_left = build_event_window_target(
        mutated,
        pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01 11:00")]
        ),
        horizon_hours=6,
    )
    assert y_left.iloc[0] == 0.0

    boundary = intervals.copy()
    boundary.loc[
        boundary["interval_start"]
        == pd.Timestamp("2020-01-01 18:00"),
        "kp",
    ] = 6.0

    # At t=12:00, t+6h=18:00 is included.
    y_right = build_event_window_target(
        boundary,
        times,
        horizon_hours=6,
    )
    assert y_right.iloc[0] == 1.0


def test_unknown_target_never_becomes_negative_due_to_missing_future_truth():
    intervals = _constant_kp_intervals(
        periods=5,
        value=1.0,
    )

    # Last interval ends 15:00. At t=13:00, target requires 14..19.
    target, audit = build_event_window_target(
        intervals,
        pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01 13:00")]
        ),
        horizon_hours=6,
        return_audit=True,
    )

    assert pd.isna(target.iloc[0])
    assert audit.iloc[0]["target_status"] == "unknown"
    assert audit.iloc[0]["missing_future_hours"] > 0


def test_known_positive_remains_positive_with_other_future_truth_missing():
    intervals = _constant_kp_intervals(
        periods=5,
        value=1.0,
    )
    intervals.loc[
        intervals["interval_start"]
        == pd.Timestamp("2020-01-01 12:00"),
        "kp",
    ] = 6.0

    target, audit = build_event_window_target(
        intervals,
        pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01 13:00")]
        ),
        horizon_hours=6,
        return_audit=True,
    )

    assert target.iloc[0] == 1.0
    assert audit.iloc[0]["target_status"] == "positive"
    assert audit.iloc[0]["missing_future_hours"] > 0


def test_calendar_split_boundary_does_not_change_feature_or_target_semantics():
    omni = _omni(
        start="2021-12-30 00:00",
        periods=120,
    )
    intervals = build_kp_intervals(
        omni[["kp_raw"]]
    )

    prediction_times = pd.DatetimeIndex(
        [
            "2021-12-31 23:00",
            "2022-01-01 00:00",
            "2022-01-01 01:00",
        ]
    )

    x, audit = build_primary_feature_frame(
        omni,
        intervals,
        prediction_times,
        return_audit=True,
    )
    y = build_event_window_target(
        intervals,
        prediction_times,
    )

    assert x.shape == (3, 93)
    assert y.index.equals(x.index)

    expected_cutoff = (
        prediction_times
        - pd.Timedelta(hours=1)
    )
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(
            audit["information_cutoff"]
        ),
        expected_cutoff,
        check_names=False,
    )

    valid = audit[
        "maximum_feature_information_time"
    ].notna()
    assert (
        audit.loc[
            valid,
            "maximum_feature_information_time",
        ]
        <= audit.loc[
            valid,
            "information_cutoff",
        ]
    ).all()


def test_prediction_grid_sparsity_does_not_change_composed_x_or_y():
    omni = _omni()
    intervals = build_kp_intervals(
        omni[["kp_raw"]]
    )

    sparse_times = pd.DatetimeIndex(
        [
            "2020-01-01 12:00",
            "2020-01-02 00:00",
        ]
    )
    single_time = pd.DatetimeIndex(
        ["2020-01-02 00:00"]
    )

    x_sparse = build_primary_feature_frame(
        omni,
        intervals,
        sparse_times,
    )
    y_sparse = build_event_window_target(
        intervals,
        sparse_times,
    )

    x_single = build_primary_feature_frame(
        omni,
        intervals,
        single_time,
    )
    y_single = build_event_window_target(
        intervals,
        single_time,
    )

    pd.testing.assert_series_equal(
        x_sparse.loc["2020-01-02 00:00"],
        x_single.loc["2020-01-02 00:00"],
        check_names=False,
    )
    assert (
        y_sparse.loc["2020-01-02 00:00"]
        == y_single.loc["2020-01-02 00:00"]
    )
