import numpy as np
import pandas as pd
import pytest

from src.data.kp import build_kp_intervals
from src.features.raw import (
    PRIMARY_RAW_FEATURE_COLUMNS,
    build_raw_features,
)


def _omni(start="2020-01-01 00:00", periods=30):
    idx = pd.date_range(start, periods=periods, freq="h")
    frame = pd.DataFrame(
        {
            "bz_gsm": np.arange(periods, dtype=float),
            "bt": np.arange(periods, dtype=float) + 100,
            "speed": np.arange(periods, dtype=float) + 400,
            "density": np.arange(periods, dtype=float) + 5,
            "flow_pressure": np.arange(periods, dtype=float) + 1,
            "kp_raw": np.repeat(
                [10, 20, 30, 40, 50, 60, 70, 80, 90, 80],
                3,
            )[:periods],
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    return frame


def test_build_raw_features_has_frozen_primary_columns():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    result = build_raw_features(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-02 00:00"]),
    )

    assert tuple(result.columns) == PRIMARY_RAW_FEATURE_COLUMNS


def test_omni_uses_exact_latest_eligible_interval():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    result, audit = build_raw_features(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        return_audit=True,
    )

    # At t=12:00, cutoff=11:00 and the latest eligible hourly OMNI
    # interval is [10:00, 11:00).
    assert result.loc["2020-01-01 12:00", "bz_gsm"] == 10.0
    assert (
        audit.loc[
            "2020-01-01 12:00",
            "omni_information_time",
        ]
        == pd.Timestamp("2020-01-01 11:00")
    )


def test_exact_cutoff_equality_is_eligible():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    _, audit = build_raw_features(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-01 08:00"]),
        return_audit=True,
    )

    assert (
        audit.loc[
            "2020-01-01 08:00",
            "omni_information_time",
        ]
        == audit.loc[
            "2020-01-01 08:00",
            "information_cutoff",
        ]
    )


def test_missing_expected_omni_timestamp_does_not_fallback():
    omni = _omni().drop(pd.Timestamp("2020-01-01 10:00"))

    # Kp intervals are built from an intact Kp source for this focused test.
    kp_source = _omni()[["kp_raw"]]
    intervals = build_kp_intervals(kp_source)

    result, audit = build_raw_features(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        return_audit=True,
    )

    assert result.loc["2020-01-01 12:00", list(
        PRIMARY_RAW_FEATURE_COLUMNS[:5]
    )].isna().all()
    assert pd.isna(
        audit.loc[
            "2020-01-01 12:00",
            "omni_information_time",
        ]
    )


def test_missing_omni_value_is_preserved():
    omni = _omni()
    omni.loc["2020-01-01 10:00", "density"] = np.nan
    intervals = build_kp_intervals(omni[["kp_raw"]])

    result = build_raw_features(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    assert pd.isna(
        result.loc["2020-01-01 12:00", "density"]
    )
    assert result.loc["2020-01-01 12:00", "speed"] == 410.0


def test_primary_omni_fill_values_become_missing():
    omni = _omni()
    omni.loc["2020-01-01 10:00", "bz_gsm"] = 999.9
    omni.loc["2020-01-01 10:00", "speed"] = 9999.0
    omni.loc["2020-01-01 10:00", "flow_pressure"] = 99.99
    intervals = build_kp_intervals(omni[["kp_raw"]])

    result = build_raw_features(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    assert pd.isna(result.loc["2020-01-01 12:00", "bz_gsm"])
    assert pd.isna(result.loc["2020-01-01 12:00", "speed"])
    assert pd.isna(
        result.loc["2020-01-01 12:00", "flow_pressure"]
    )


def test_kp_lags_use_canonical_asof_semantics():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    result, audit = build_raw_features(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        return_audit=True,
    )

    # t-1h = 11:00 -> most recent completed Kp interval ends 09:00,
    # whose value is 3.0.
    assert result.loc["2020-01-01 12:00", "kp_lag_1h"] == 3.0
    assert (
        audit.loc[
            "2020-01-01 12:00",
            "kp_lag_1h_information_time",
        ]
        == pd.Timestamp("2020-01-01 09:00")
    )


def test_all_feature_provenance_respects_cutoff():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    _, audit = build_raw_features(
        omni,
        intervals,
        pd.date_range("2020-01-01 06:00", periods=10, freq="h"),
        return_audit=True,
    )

    valid = audit["maximum_feature_information_time"].notna()
    assert (
        audit.loc[valid, "maximum_feature_information_time"]
        <= audit.loc[valid, "information_cutoff"]
    ).all()


def test_future_omni_mutation_does_not_change_past_features():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])
    times = pd.DatetimeIndex(["2020-01-01 12:00"])

    before = build_raw_features(omni, intervals, times)

    mutated = omni.copy()
    mutated.loc[
        mutated.index >= pd.Timestamp("2020-01-01 11:00"),
        ["bz_gsm", "bt", "speed", "density", "flow_pressure"],
    ] = -123456.0

    after = build_raw_features(mutated, intervals, times)

    pd.testing.assert_frame_equal(before, after)


def test_prediction_rows_are_not_dropped_for_missing_features():
    omni = _omni()
    omni.loc["2020-01-01 10:00", "density"] = np.nan
    intervals = build_kp_intervals(omni[["kp_raw"]])
    times = pd.DatetimeIndex(
        ["2020-01-01 02:00", "2020-01-01 12:00"]
    )

    result = build_raw_features(omni, intervals, times)

    assert result.index.equals(
        pd.DatetimeIndex(times, name="prediction_time")
    )
    assert len(result) == 2


@pytest.mark.parametrize(
    "times",
    [
        ["2020-01-01 12:30"],
        ["2020-01-01 13:00", "2020-01-01 12:00"],
        ["2020-01-01 12:00", "2020-01-01 12:00"],
    ],
)
def test_invalid_prediction_times_raise(times):
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    with pytest.raises(ValueError):
        build_raw_features(omni, intervals, pd.DatetimeIndex(times))
