import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.definitions.events import identify_events
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS
from src.phase7.contract import (
    PHASE7_EXPERIMENTS,
    PHASE7_EXPERIMENT_IDS,
    PHASE7_FEATURES,
    PHASE7_IMBALANCE_EXPERIMENT,
    PHASE7_MODEL_CONFIG_ID,
    PHASE7_PRIMARY_CONTROL_ID,
    PHASE7_TERMINATION_HOURS,
    build_phase7_events,
    build_phase7_target,
    get_phase7_experiment,
    validate_phase7_contract,
)
from src.targets.event_window import build_event_window_target


def _kp_intervals() -> pd.DataFrame:
    starts = pd.date_range(
        "2020-01-01 00:00",
        periods=12,
        freq="3h",
    )
    values = [
        2.0, 2.0, 5.0, 5.0,
        2.0, 2.0, 6.0, 6.0,
        2.0, 2.0, 7.0, 2.0,
    ]
    return pd.DataFrame(
        {
            "interval_start": starts,
            "interval_end": starts + pd.Timedelta(hours=3),
            "kp": values,
        }
    )


def _omni() -> pd.DataFrame:
    index = pd.date_range(
        "2019-12-30 00:00",
        periods=120,
        freq="h",
        name="timestamp",
    )
    x = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "bz_gsm": -5.0 + 0.01 * x,
            "bt": 7.0 + 0.02 * x,
            "speed": 400.0 + x,
            "density": 5.0 + 0.01 * x,
            "flow_pressure": 1.5 + 0.001 * x,
        },
        index=index,
    )


def test_phase7_registry_is_exactly_frozen():
    validate_phase7_contract()

    assert PHASE7_EXPERIMENT_IDS == (
        "t5_h3",
        "t5_h6",
        "t5_h12",
        "t5_h24",
        "t6_h6",
        "t7_h6",
    )

    assert tuple(
        (x.threshold, x.horizon_hours)
        for x in PHASE7_EXPERIMENTS
    ) == (
        (5.0, 3),
        (5.0, 6),
        (5.0, 12),
        (5.0, 24),
        (6.0, 6),
        (7.0, 6),
    )


def test_phase7_freezes_primary_model_inputs_and_imbalance():
    assert PHASE7_FEATURES == tuple(PRIMARY_RAW_FEATURE_COLUMNS)
    assert len(PHASE7_FEATURES) == 10
    assert PHASE7_IMBALANCE_EXPERIMENT == "none"
    assert PHASE7_MODEL_CONFIG_ID == "lightgbm_lr0.1_leaves127"

    control = get_phase7_experiment(PHASE7_PRIMARY_CONTROL_ID)
    assert control.threshold == 5.0
    assert control.horizon_hours == 6
    assert control.is_primary_control


def test_phase7_target_wrapper_is_exact_canonical_parameterization():
    kp = _kp_intervals()
    times = pd.date_range(
        "2020-01-01 03:00",
        periods=20,
        freq="h",
    )

    for experiment in PHASE7_EXPERIMENTS:
        actual, actual_audit = build_phase7_target(
            kp,
            times,
            experiment,
            return_audit=True,
        )
        expected, expected_audit = build_event_window_target(
            kp,
            times,
            threshold=experiment.threshold,
            horizon_hours=experiment.horizon_hours,
            return_audit=True,
        )

        assert_series_equal(actual, expected)
        assert_frame_equal(actual_audit, expected_audit)


def test_horizon_changes_target_but_not_event_truth():
    kp = _kp_intervals()
    times = pd.DatetimeIndex(
        ["2020-01-01 02:00"]
    )

    h3 = build_phase7_target(
        kp,
        times,
        "t5_h3",
    )
    h6 = build_phase7_target(
        kp,
        times,
        "t5_h6",
    )

    assert h3.iloc[0] == 0.0
    assert h6.iloc[0] == 1.0

    events_h3 = build_phase7_events(
        kp,
        "t5_h3",
    )
    events_h6 = build_phase7_events(
        kp,
        "t5_h6",
    )

    assert_frame_equal(
        events_h3,
        events_h6,
    )


def test_severity_changes_target_and_event_truth_when_kp_supports_it():
    kp = _kp_intervals()
    times = pd.DatetimeIndex(
        ["2020-01-01 15:00"]
    )

    t5 = build_phase7_target(kp, times, "t5_h6")
    t7 = build_phase7_target(kp, times, "t7_h6")

    assert t5.iloc[0] == 1.0
    assert t7.iloc[0] == 0.0

    events_t5 = build_phase7_events(kp, "t5_h6")
    events_t7 = build_phase7_events(kp, "t7_h6")

    assert len(events_t5) > len(events_t7)
    assert (events_t5["threshold"] == 5.0).all()
    assert (events_t7["threshold"] == 7.0).all()


def test_phase7_event_wrapper_keeps_frozen_termination_semantics():
    kp = _kp_intervals()

    for experiment in PHASE7_EXPERIMENTS:
        actual = build_phase7_events(kp, experiment)
        expected = identify_events(
            kp,
            threshold=experiment.threshold,
            termination_hours=PHASE7_TERMINATION_HOURS,
        )
        assert_frame_equal(actual, expected)


def test_unknown_experiment_id_is_rejected():
    try:
        get_phase7_experiment("t8_h48")
    except KeyError as exc:
        assert "Unknown Phase 7 experiment" in str(exc)
    else:
        raise AssertionError("Unregistered Phase 7 experiment was accepted.")
