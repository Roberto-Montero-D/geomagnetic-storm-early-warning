import numpy as np
import pandas as pd

from src.features.dynamics import (
    DYNAMIC_FEATURE_COLUMNS,
    build_dynamic_features,
)


def _omni(periods=20):
    idx = pd.date_range("2020-01-01 00:00", periods=periods, freq="h")
    x = np.arange(periods, dtype=float)

    frame = pd.DataFrame(
        {
            "bz_gsm": x,
            "bt": x * 2.0,
            "speed": 400.0 + x * 3.0,
            "density": 5.0 + x * 0.5,
            "flow_pressure": 1.0 + x * 0.25,
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    return frame


def test_dynamic_columns_are_frozen():
    result = build_dynamic_features(
        _omni(),
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )
    assert tuple(result.columns) == DYNAMIC_FEATURE_COLUMNS
    assert len(result.columns) == 15


def test_one_and_three_hour_deltas_use_exact_timestamps():
    result = build_dynamic_features(
        _omni(),
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    # Latest eligible source start is 10:00.
    assert result.iloc[0]["bz_gsm_delta_1h"] == 1.0
    assert result.iloc[0]["bz_gsm_delta_3h"] == 3.0


def test_three_hour_slope_is_ols_units_per_hour():
    result = build_dynamic_features(
        _omni(),
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    assert result.iloc[0]["bz_gsm_slope_3h"] == 1.0
    assert result.iloc[0]["bt_slope_3h"] == 2.0
    assert result.iloc[0]["speed_slope_3h"] == 3.0


def test_missing_required_timestamp_makes_affected_dynamics_nan():
    omni = _omni().drop(pd.Timestamp("2020-01-01 09:00"))

    result = build_dynamic_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    # delta_1h requires exact 09:00 -> missing.
    assert pd.isna(result.iloc[0]["bz_gsm_delta_1h"])
    # delta_3h uses 10:00 and 07:00, which still exist.
    assert result.iloc[0]["bz_gsm_delta_3h"] == 3.0
    # slope requires 07,08,09,10 -> missing.
    assert pd.isna(result.iloc[0]["bz_gsm_slope_3h"])


def test_missing_value_makes_only_affected_variable_features_nan():
    omni = _omni()
    omni.loc["2020-01-01 09:00", "bz_gsm"] = np.nan

    result = build_dynamic_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    assert pd.isna(result.iloc[0]["bz_gsm_delta_1h"])
    assert pd.isna(result.iloc[0]["bz_gsm_slope_3h"])
    assert result.iloc[0]["speed_delta_1h"] == 3.0


def test_fill_value_is_treated_as_missing():
    omni = _omni()
    omni.loc["2020-01-01 09:00", "speed"] = 9999.0

    result = build_dynamic_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    assert pd.isna(result.iloc[0]["speed_delta_1h"])
    assert pd.isna(result.iloc[0]["speed_slope_3h"])


def test_future_mutation_does_not_change_dynamics():
    omni = _omni()
    times = pd.DatetimeIndex(["2020-01-01 12:00"])

    before = build_dynamic_features(omni, times)

    mutated = omni.copy()
    mutated.loc[
        mutated.index >= pd.Timestamp("2020-01-01 11:00"),
        :,
    ] = -123456.0

    after = build_dynamic_features(mutated, times)

    pd.testing.assert_frame_equal(before, after)


def test_dynamic_provenance_respects_cutoff():
    _, audit = build_dynamic_features(
        _omni(),
        pd.date_range("2020-01-01 05:00", periods=10, freq="h"),
        return_audit=True,
    )

    valid = audit["dynamics_information_time"].notna()
    assert (
        audit.loc[valid, "dynamics_information_time"]
        <= audit.loc[valid, "information_cutoff"]
    ).all()


def test_sparse_prediction_grid_does_not_change_dynamics():
    omni = _omni()

    sparse = build_dynamic_features(
        omni,
        pd.DatetimeIndex(
            ["2020-01-01 06:00", "2020-01-01 12:00"]
        ),
    )
    single = build_dynamic_features(
        omni,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    pd.testing.assert_series_equal(
        sparse.loc["2020-01-01 12:00"],
        single.loc["2020-01-01 12:00"],
        check_names=False,
    )
