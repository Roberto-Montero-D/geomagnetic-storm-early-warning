import numpy as np
import pandas as pd

from src.features.persistence import (
    PERSISTENCE_FEATURE_COLUMNS,
    build_persistence_features,
)


def _omni(periods=20):
    idx = pd.date_range("2020-01-01 00:00", periods=periods, freq="h")
    frame = pd.DataFrame(
        {
            "bz_gsm": np.zeros(periods, dtype=float),
            "speed": np.full(periods, 400.0),
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    return frame


def test_persistence_columns_are_frozen():
    result = build_persistence_features(
        _omni(),
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )
    assert tuple(result.columns) == PERSISTENCE_FEATURE_COLUMNS


def test_bz_persistence_counts_consecutive_latest_eligible_hours():
    omni = _omni()
    omni.loc["2020-01-01 07:00":"2020-01-01 10:00", "bz_gsm"] = -6.0

    result = build_persistence_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    # Latest eligible start is 10:00; 07,08,09,10 all satisfy Bz < -5.
    assert result.iloc[0]["bz_gsm_persist_lt_m5h"] == 4.0
    assert result.iloc[0]["bz_gsm_persist_lt_m10h"] == 0.0


def test_speed_persistence_thresholds_are_strict():
    omni = _omni()
    omni.loc["2020-01-01 09:00":"2020-01-01 10:00", "speed"] = 500.0

    result = build_persistence_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    assert result.iloc[0]["speed_persist_gt_500h"] == 0.0


def test_missing_timestamp_breaks_persistence_run():
    omni = _omni()
    omni.loc["2020-01-01 06:00":"2020-01-01 10:00", "bz_gsm"] = -8.0
    omni = omni.drop(pd.Timestamp("2020-01-01 08:00"))

    result = build_persistence_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    # 10 and 09 satisfy; 08 is absent and breaks the run.
    assert result.iloc[0]["bz_gsm_persist_lt_m5h"] == 2.0


def test_missing_latest_timestamp_yields_nan_not_zero():
    omni = _omni().drop(pd.Timestamp("2020-01-01 10:00"))

    result = build_persistence_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    assert result.iloc[0].isna().all()


def test_missing_latest_value_yields_nan_for_affected_variable():
    omni = _omni()
    omni.loc["2020-01-01 10:00", "bz_gsm"] = np.nan

    result = build_persistence_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    assert pd.isna(result.iloc[0]["bz_gsm_persist_lt_m5h"])
    assert result.iloc[0]["speed_persist_gt_500h"] == 0.0


def test_fill_value_breaks_run_as_missing():
    omni = _omni()
    omni.loc["2020-01-01 08:00":"2020-01-01 10:00", "bz_gsm"] = -8.0
    omni.loc["2020-01-01 09:00", "bz_gsm"] = 999.9

    result = build_persistence_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    assert result.iloc[0]["bz_gsm_persist_lt_m5h"] == 1.0


def test_future_mutation_does_not_change_persistence():
    omni = _omni()
    omni.loc["2020-01-01 08:00":"2020-01-01 10:00", "bz_gsm"] = -8.0

    times = pd.DatetimeIndex(["2020-01-01 12:00"])
    before = build_persistence_features(omni, times)

    mutated = omni.copy()
    mutated.loc[
        mutated.index >= pd.Timestamp("2020-01-01 11:00"),
        ["bz_gsm", "speed"],
    ] = -999999.0

    after = build_persistence_features(mutated, times)

    pd.testing.assert_frame_equal(before, after)


def test_persistence_provenance_respects_cutoff():
    _, audit = build_persistence_features(
        _omni(),
        pd.date_range("2020-01-01 03:00", periods=10, freq="h"),
        return_audit=True,
    )

    valid = audit["persistence_information_time"].notna()
    assert (
        audit.loc[valid, "persistence_information_time"]
        <= audit.loc[valid, "information_cutoff"]
    ).all()
