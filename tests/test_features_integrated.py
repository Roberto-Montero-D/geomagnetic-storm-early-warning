import numpy as np
import pandas as pd

from src.data.kp import build_kp_intervals
from src.features.integrated import (
    FEATURE_FAMILY_ORDER,
    PRIMARY_FEATURE_COLUMNS,
    build_primary_feature_frame,
)
from src.features.interactions import INTERACTION_FEATURE_COLUMNS


def _omni(periods=48):
    idx = pd.date_range("2020-01-01 00:00", periods=periods, freq="h")
    x = np.arange(periods, dtype=float)

    frame = pd.DataFrame(
        {
            "bz_gsm": -2.0 - (x % 8),
            "bt": 5.0 + x * 0.1,
            "speed": 400.0 + x * 2.0,
            "density": 5.0 + x * 0.05,
            "flow_pressure": 1.0 + x * 0.02,
            "kp_raw": np.resize(
                np.repeat([10, 20, 30, 40, 50, 30, 20, 10], 3),
                periods,
            ),
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    return frame


def test_manifest_family_order_is_frozen():
    assert FEATURE_FAMILY_ORDER == (
        "raw",
        "rolling",
        "persistence",
        "dynamics",
        "interactions",
    )


def test_manifest_has_expected_count_and_unique_names():
    # raw=10, rolling=60, persistence=5, dynamics=15, interactions=3
    assert len(PRIMARY_FEATURE_COLUMNS) == 93
    assert len(set(PRIMARY_FEATURE_COLUMNS)) == 93


def test_integrated_frame_uses_exact_manifest_order():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    result = build_primary_feature_frame(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-02 06:00"]),
    )

    assert tuple(result.columns) == PRIMARY_FEATURE_COLUMNS
    assert not result.columns.has_duplicates


def test_interactions_equal_transform_of_integrated_raw_state():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    result = build_primary_feature_frame(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-01 12:00"]),
    )

    row = result.iloc[0]
    bz_neg = max(-row["bz_gsm"], 0.0)

    assert row["bz_neg_x_speed"] == bz_neg * row["speed"]
    assert row["bz_neg_x_density"] == bz_neg * row["density"]
    assert (
        row["flow_pressure_x_speed"]
        == row["flow_pressure"] * row["speed"]
    )


def test_integrated_provenance_respects_information_cutoff():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])

    _, audit = build_primary_feature_frame(
        omni,
        intervals,
        pd.date_range("2020-01-01 08:00", periods=20, freq="h"),
        return_audit=True,
    )

    valid = audit["maximum_feature_information_time"].notna()
    assert (
        audit.loc[valid, "maximum_feature_information_time"]
        <= audit.loc[valid, "information_cutoff"]
    ).all()


def test_future_omni_mutation_cannot_change_complete_past_feature_vector():
    omni = _omni()
    intervals = build_kp_intervals(omni[["kp_raw"]])
    times = pd.DatetimeIndex(["2020-01-01 18:00"])

    before = build_primary_feature_frame(omni, intervals, times)

    mutated = omni.copy()
    # At prediction 18:00, information cutoff is 17:00. OMNI source rows
    # beginning at 17:00 or later are unavailable and must be irrelevant.
    mutated.loc[
        mutated.index >= pd.Timestamp("2020-01-01 17:00"),
        ["bz_gsm", "bt", "speed", "density", "flow_pressure"],
    ] = -123456.0

    after = build_primary_feature_frame(mutated, intervals, times)

    pd.testing.assert_frame_equal(before, after)


def test_future_kp_mutation_cannot_change_complete_past_feature_vector():
    omni = _omni()
    times = pd.DatetimeIndex(["2020-01-01 18:00"])
    intervals_before = build_kp_intervals(omni[["kp_raw"]])

    before = build_primary_feature_frame(
        omni,
        intervals_before,
        times,
    )

    mutated = omni.copy()
    mutated.loc[
        mutated.index >= pd.Timestamp("2020-01-01 18:00"),
        "kp_raw",
    ] = 90
    intervals_after = build_kp_intervals(mutated[["kp_raw"]])

    after = build_primary_feature_frame(
        mutated,
        intervals_after,
        times,
    )

    pd.testing.assert_frame_equal(before, after)


def test_missing_latest_omni_timestamp_is_not_substituted_cross_layer():
    omni = _omni().drop(pd.Timestamp("2020-01-01 16:00"))
    kp_source = _omni()[["kp_raw"]]
    intervals = build_kp_intervals(kp_source)

    result = build_primary_feature_frame(
        omni,
        intervals,
        pd.DatetimeIndex(["2020-01-01 18:00"]),
    )

    # Exact latest eligible source start is 16:00.
    for column in (
        "bz_gsm",
        "bt",
        "speed",
        "density",
        "flow_pressure",
        *INTERACTION_FEATURE_COLUMNS,
    ):
        assert pd.isna(result.iloc[0][column])

    # Dynamics requiring the missing latest state must also be NaN.
    assert pd.isna(result.iloc[0]["bz_gsm_delta_1h"])
    assert pd.isna(result.iloc[0]["bz_gsm_delta_3h"])
    assert pd.isna(result.iloc[0]["bz_gsm_slope_3h"])

    # Persistence latest state is unknown, not false.
    assert pd.isna(result.iloc[0]["bz_gsm_persist_lt_m5h"])

    # Rolling windows may still use other physically present observations
    # inside the fixed window; they must not extend the window backward.
    assert pd.notna(result.iloc[0]["bz_gsm_roll_mean_3h"])


def test_missing_feature_values_do_not_drop_prediction_rows():
    omni = _omni()
    omni.loc["2020-01-01 16:00", "density"] = np.nan
    intervals = build_kp_intervals(omni[["kp_raw"]])
    times = pd.DatetimeIndex(
        ["2020-01-01 03:00", "2020-01-01 18:00"]
    )

    result = build_primary_feature_frame(omni, intervals, times)

    assert len(result) == 2
    assert result.index.equals(
        pd.DatetimeIndex(times, name="prediction_time")
    )
