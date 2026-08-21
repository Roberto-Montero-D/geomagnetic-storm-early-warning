import numpy as np
import pandas as pd
import pytest

from src.features.rolling import (
    ROLLING_WINDOWS_HOURS,
    build_rolling_features,
    rolling_feature_names,
)


def _omni(start="2020-01-01 00:00", periods=40):
    idx = pd.date_range(start, periods=periods, freq="h")
    x = np.arange(periods, dtype=float)
    frame = pd.DataFrame(
        {
            "bz_gsm": x,
            "bt": x + 100.0,
            "speed": x + 400.0,
            "density": x + 5.0,
            "flow_pressure": x + 1.0,
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    return frame


def test_frozen_rolling_feature_columns():
    result = build_rolling_features(
        _omni(),
        pd.DatetimeIndex(["2020-01-02 06:00"]),
    )
    assert tuple(result.columns) == rolling_feature_names()
    assert len(result.columns) == 5 * 4 * 3
    assert ROLLING_WINDOWS_HOURS == (3, 6, 12, 24)


def test_three_hour_window_uses_physical_interval_ends():
    omni = _omni()
    result, audit = build_rolling_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        return_audit=True,
    )

    # cutoff=11:00. 3h window is (08:00, 11:00] by interval end,
    # therefore source starts 08:00, 09:00, 10:00 -> values 8,9,10.
    row = result.loc["2020-01-01 12:00"]
    assert row["bz_gsm_roll_mean_3h"] == 9.0
    assert row["bz_gsm_roll_min_3h"] == 8.0
    assert row["bz_gsm_roll_std_3h"] == 1.0
    assert (
        audit.loc[
            "2020-01-01 12:00",
            "maximum_rolling_information_time",
        ]
        == pd.Timestamp("2020-01-01 11:00")
    )


def test_window_left_boundary_is_excluded_and_cutoff_is_included():
    omni = _omni()
    result = build_rolling_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        windows_hours=(3,),
    )
    # Source start 07:00 has end 08:00 and is excluded.
    # Source start 10:00 has end 11:00 and is included.
    assert result.iloc[0]["bz_gsm_roll_min_3h"] == 8.0
    assert result.iloc[0]["bz_gsm_roll_mean_3h"] == 9.0


def test_missing_timestamp_does_not_extend_window_backward():
    omni = _omni().drop(pd.Timestamp("2020-01-01 09:00"))
    result = build_rolling_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        windows_hours=(3,),
    )
    # Physical window still contains starts 08,09,10; 09 is absent.
    # It must not reach back to 07 to collect a third row.
    assert result.iloc[0]["bz_gsm_roll_mean_3h"] == 9.0
    assert result.iloc[0]["bz_gsm_roll_min_3h"] == 8.0


def test_missing_value_is_ignored_without_imputation():
    omni = _omni()
    omni.loc["2020-01-01 09:00", "bz_gsm"] = np.nan
    result = build_rolling_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        windows_hours=(3,),
    )
    assert result.iloc[0]["bz_gsm_roll_mean_3h"] == 9.0
    assert result.iloc[0]["bz_gsm_roll_min_3h"] == 8.0


def test_fill_value_is_treated_as_missing():
    omni = _omni()
    omni.loc["2020-01-01 09:00", "bz_gsm"] = 999.9
    result = build_rolling_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        windows_hours=(3,),
    )
    assert result.iloc[0]["bz_gsm_roll_mean_3h"] == 9.0


def test_std_is_nan_with_only_one_valid_value():
    omni = _omni()
    omni.loc["2020-01-01 08:00", "bz_gsm"] = np.nan
    omni.loc["2020-01-01 09:00", "bz_gsm"] = np.nan
    result = build_rolling_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        windows_hours=(3,),
    )
    assert result.iloc[0]["bz_gsm_roll_mean_3h"] == 10.0
    assert pd.isna(result.iloc[0]["bz_gsm_roll_std_3h"])


def test_no_valid_values_yields_nan():
    omni = _omni()
    omni.loc[
        pd.date_range("2020-01-01 08:00", periods=3, freq="h"),
        "bz_gsm",
    ] = np.nan
    result = build_rolling_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        windows_hours=(3,),
    )
    assert pd.isna(result.iloc[0]["bz_gsm_roll_mean_3h"])
    assert pd.isna(result.iloc[0]["bz_gsm_roll_min_3h"])
    assert pd.isna(result.iloc[0]["bz_gsm_roll_std_3h"])


def test_future_mutation_does_not_change_past_rolling_features():
    omni = _omni()
    times = pd.DatetimeIndex(["2020-01-01 12:00"])
    before = build_rolling_features(omni, times)

    mutated = omni.copy()
    mutated.loc[
        mutated.index >= pd.Timestamp("2020-01-01 11:00"),
        :,
    ] = -123456.0
    after = build_rolling_features(mutated, times)

    pd.testing.assert_frame_equal(before, after)


def test_all_rolling_provenance_respects_cutoff():
    _, audit = build_rolling_features(
        _omni(),
        pd.date_range("2020-01-01 06:00", periods=20, freq="h"),
        return_audit=True,
    )
    valid = audit["maximum_rolling_information_time"].notna()
    assert (
        audit.loc[valid, "maximum_rolling_information_time"]
        <= audit.loc[valid, "information_cutoff"]
    ).all()


def test_sparse_prediction_times_do_not_change_window_definition():
    omni = _omni()
    sparse = build_rolling_features(
        omni,
        pd.DatetimeIndex(
            ["2020-01-01 06:00", "2020-01-01 12:00"]
        ),
        windows_hours=(3,),
    )
    single = build_rolling_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
        windows_hours=(3,),
    )
    pd.testing.assert_series_equal(
        sparse.loc["2020-01-01 12:00"],
        single.loc["2020-01-01 12:00"],
        check_names=False,
    )


@pytest.mark.parametrize(
    "times",
    [
        ["2020-01-01 12:30"],
        ["2020-01-01 13:00", "2020-01-01 12:00"],
        ["2020-01-01 12:00", "2020-01-01 12:00"],
    ],
)
def test_invalid_prediction_times_raise(times):
    with pytest.raises(ValueError):
        build_rolling_features(_omni(), pd.DatetimeIndex(times))


@pytest.mark.parametrize(
    "windows",
    [
        (),
        (0,),
        (-3,),
        (3, 3),
        (3.0,),
        (True,),
    ],
)
def test_invalid_windows_raise(windows):
    with pytest.raises(ValueError):
        build_rolling_features(
            _omni(),
            pd.DatetimeIndex(["2020-01-01 12:00"]),
            windows_hours=windows,
        )
