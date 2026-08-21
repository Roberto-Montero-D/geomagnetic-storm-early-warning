import numpy as np
import pandas as pd
import pytest

from src.targets.event_window import build_event_window_target


def _intervals(
    values,
    *,
    start="2020-01-01 00:00",
):
    starts = pd.date_range(
        start,
        periods=len(values),
        freq="3h",
    )
    return pd.DataFrame(
        {
            "interval_start": starts,
            "interval_end": starts + pd.Timedelta(hours=3),
            "kp": values,
        }
    )


def test_primary_target_positive_when_future_storm_condition_exists():
    kp = _intervals([1.0, 5.0, 2.0])

    target = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 02:00"]),
    )

    # Future window is 03:00..08:00 and includes Kp=5 at 03:00.
    assert target.iloc[0] == 1.0


def test_target_uses_storm_conditions_not_only_event_start():
    kp = _intervals([1.0, 6.0, 2.0])

    target = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 03:00"]),
    )

    # Storm interval already began at 03:00, but 04:00 and 05:00 are future
    # storm conditions, so the target remains positive.
    assert target.iloc[0] == 1.0


def test_current_time_is_excluded_from_future_window():
    kp = _intervals([6.0, 1.0, 1.0, 1.0])

    target = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 02:00"]),
    )

    # Kp at t=02:00 is storm-level, but t itself is excluded.
    # Future 03:00..08:00 is all below threshold.
    assert target.iloc[0] == 0.0


def test_horizon_right_boundary_is_inclusive():
    kp = _intervals([1.0, 1.0, 5.0, 1.0])

    target = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 00:00"]),
        horizon_hours=6,
    )

    # t+6h = 06:00 is included and is storm-level.
    assert target.iloc[0] == 1.0


def test_state_after_horizon_is_excluded():
    kp = _intervals([1.0, 1.0, 1.0, 5.0])

    target = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 00:00"]),
        horizon_hours=6,
    )

    assert target.iloc[0] == 0.0


def test_complete_below_threshold_horizon_is_negative():
    kp = _intervals([1.0, 2.0, 3.0, 4.0])

    target, audit = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 01:00"]),
        return_audit=True,
    )

    assert target.iloc[0] == 0.0
    assert audit.iloc[0]["target_status"] == "negative"
    assert audit.iloc[0]["missing_future_hours"] == 0


def test_missing_future_kp_without_positive_is_unknown():
    kp = _intervals([1.0, np.nan, 2.0, 2.0])

    target, audit = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 01:00"]),
        return_audit=True,
    )

    assert pd.isna(target.iloc[0])
    assert audit.iloc[0]["target_status"] == "unknown"
    assert audit.iloc[0]["missing_future_hours"] > 0


def test_known_positive_overrides_other_missing_future_hours():
    kp = _intervals([1.0, np.nan, 5.0, 1.0])

    target, audit = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 01:00"]),
        return_audit=True,
    )

    assert target.iloc[0] == 1.0
    assert audit.iloc[0]["target_status"] == "positive"
    assert audit.iloc[0]["missing_future_hours"] > 0


def test_incomplete_right_edge_horizon_is_unknown_when_no_positive():
    kp = _intervals([1.0, 1.0])

    target, audit = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 04:00"]),
        horizon_hours=6,
        return_audit=True,
    )

    assert pd.isna(target.iloc[0])
    assert audit.iloc[0]["target_status"] == "unknown"
    assert audit.iloc[0]["observed_future_hours"] == 1
    assert audit.iloc[0]["missing_future_hours"] == 5


def test_incomplete_right_edge_can_still_be_known_positive():
    kp = _intervals([1.0, 5.0])

    target = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 04:00"]),
        horizon_hours=6,
    )

    # 05:00 is observed storm-level; existential condition is already known.
    assert target.iloc[0] == 1.0


def test_missing_canonical_interval_is_unknown_not_quiet():
    kp = _intervals([1.0, 2.0, 2.0, 2.0]).drop(
        index=1
    ).reset_index(drop=True)

    target = build_event_window_target(
        kp,
        pd.DatetimeIndex(["2020-01-01 01:00"]),
    )

    assert pd.isna(target.iloc[0])


def test_past_kp_mutation_does_not_change_target():
    kp = _intervals([1.0, 2.0, 5.0, 1.0])
    times = pd.DatetimeIndex(["2020-01-01 03:00"])

    before = build_event_window_target(kp, times)

    mutated = kp.copy()
    mutated.loc[
        mutated["interval_end"] <= pd.Timestamp("2020-01-01 03:00"),
        "kp",
    ] = 9.0

    after = build_event_window_target(mutated, times)

    pd.testing.assert_series_equal(before, after)


def test_kp_beyond_horizon_does_not_change_target():
    kp = _intervals([1.0, 1.0, 1.0, 1.0, 1.0])
    times = pd.DatetimeIndex(["2020-01-01 00:00"])

    before = build_event_window_target(kp, times)

    mutated = kp.copy()
    mutated.loc[
        mutated["interval_start"] >= pd.Timestamp("2020-01-01 09:00"),
        "kp",
    ] = 9.0

    after = build_event_window_target(mutated, times)

    pd.testing.assert_series_equal(before, after)


def test_malformed_kp_raises():
    kp = pd.DataFrame(
        {
            "interval_start": pd.DatetimeIndex(
                ["2020-01-01 00:00"]
            ),
            "interval_end": pd.DatetimeIndex(
                ["2020-01-01 03:00"]
            ),
            "kp": ["invalid"],
        }
    )

    with pytest.raises((ValueError, TypeError)):
        build_event_window_target(
            kp,
            pd.DatetimeIndex(["2020-01-01 00:00"]),
        )


def test_overlapping_intervals_raise():
    kp = _intervals([1.0, 2.0])
    kp.loc[1, "interval_start"] = pd.Timestamp(
        "2020-01-01 02:00"
    )
    kp.loc[1, "interval_end"] = pd.Timestamp(
        "2020-01-01 05:00"
    )

    with pytest.raises(ValueError, match="must not overlap"):
        build_event_window_target(
            kp,
            pd.DatetimeIndex(["2020-01-01 00:00"]),
        )


def test_invalid_interval_duration_raises():
    kp = _intervals([1.0])
    kp.loc[0, "interval_end"] = pd.Timestamp(
        "2020-01-01 02:00"
    )

    with pytest.raises(ValueError, match="exactly 3 hours"):
        build_event_window_target(
            kp,
            pd.DatetimeIndex(["2020-01-01 00:00"]),
        )


@pytest.mark.parametrize(
    "times",
    [
        ["2020-01-01 00:30"],
        ["2020-01-01 01:00", "2020-01-01 00:00"],
        ["2020-01-01 00:00", "2020-01-01 00:00"],
    ],
)
def test_invalid_prediction_times_raise(times):
    with pytest.raises(ValueError):
        build_event_window_target(
            _intervals([1.0, 1.0, 1.0]),
            pd.DatetimeIndex(times),
        )


@pytest.mark.parametrize(
    ("threshold", "horizon"),
    [
        (np.nan, 6),
        (5.0, 0),
        (5.0, -1),
        (5.0, 3.0),
        (5.0, True),
    ],
)
def test_invalid_parameters_raise(threshold, horizon):
    with pytest.raises((TypeError, ValueError)):
        build_event_window_target(
            _intervals([1.0, 1.0, 1.0]),
            pd.DatetimeIndex(["2020-01-01 00:00"]),
            threshold=threshold,
            horizon_hours=horizon,
        )
