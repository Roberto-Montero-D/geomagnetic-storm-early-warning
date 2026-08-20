import numpy as np
import pandas as pd
import pytest

from src.data.kp import (
    build_kp_intervals,
    build_kp_lag_features,
    convert_omni_kp_raw,
    kp_asof,
)


def make_hourly_kp(
    start: str,
    raw_values: list[int],
) -> pd.DataFrame:
    index = pd.date_range(
        start=start,
        periods=len(raw_values),
        freq="h",
    )

    return pd.DataFrame(
        {"kp_raw": raw_values},
        index=index,
    )


def test_kp_raw_encoding_conversion():
    raw = pd.Series([0, 3, 7, 10, 50, 90, 99])

    result = convert_omni_kp_raw(raw)

    expected = pd.Series(
        [0.0, 0.3, 0.7, 1.0, 5.0, 9.0, np.nan]
    )

    pd.testing.assert_series_equal(result, expected)


def test_invalid_kp_code_raises():
    raw = pd.Series([0, 10, 55])

    with pytest.raises(ValueError, match="Invalid OMNI Kp"):
        convert_omni_kp_raw(raw)


def test_kp_three_hour_bin_structure():
    df = make_hourly_kp(
        "2020-01-01 00:00",
        [20, 20, 20, 60, 60, 60],
    )

    result = build_kp_intervals(df)

    assert len(result) == 2

    assert result.loc[0, "interval_start"] == pd.Timestamp(
        "2020-01-01 00:00"
    )
    assert result.loc[0, "interval_end"] == pd.Timestamp(
        "2020-01-01 03:00"
    )
    assert result.loc[0, "kp"] == 2.0

    assert result.loc[1, "interval_start"] == pd.Timestamp(
        "2020-01-01 03:00"
    )
    assert result.loc[1, "interval_end"] == pd.Timestamp(
        "2020-01-01 06:00"
    )
    assert result.loc[1, "kp"] == 6.0


def test_inconsistent_kp_bin_raises():
    df = make_hourly_kp(
        "2020-01-01 00:00",
        [60, 60, 50],
    )

    with pytest.raises(
        ValueError,
        match="Inconsistent repeated Kp",
    ):
        build_kp_intervals(df)


def test_incomplete_kp_bin_raises():
    df = make_hourly_kp(
        "2020-01-01 00:00",
        [60, 60],
    )

    with pytest.raises(
        ValueError,
        match="Incomplete or non-contiguous",
    ):
        build_kp_intervals(df)


def test_missing_kp_interval_remains_missing():
    df = make_hourly_kp(
        "2020-01-01 00:00",
        [99, 99, 99],
    )

    result = build_kp_intervals(df)

    assert np.isnan(result.loc[0, "kp"])


def test_partially_missing_kp_interval_raises():
    df = make_hourly_kp(
        "2020-01-01 00:00",
        [60, 99, 60],
    )

    with pytest.raises(
        ValueError,
        match="Partially missing",
    ):
        build_kp_intervals(df)


def test_kp_asof_before_first_completed_interval():
    df = make_hourly_kp(
        "2020-01-01 00:00",
        [60, 60, 60],
    )

    intervals = build_kp_intervals(df)

    result = kp_asof(
        intervals,
        [pd.Timestamp("2020-01-01 02:00")],
    )

    assert np.isnan(result.iloc[0])


def test_kp_asof_at_interval_end():
    df = make_hourly_kp(
        "2020-01-01 00:00",
        [60, 60, 60],
    )

    intervals = build_kp_intervals(df)

    result = kp_asof(
        intervals,
        [pd.Timestamp("2020-01-01 03:00")],
    )

    assert result.iloc[0] == 6.0


def test_kp_lag_1h_respects_protocol_cutoff():
    df = make_hourly_kp(
        "2020-01-01 00:00",
        [
            60, 60, 60,  # 00-03
            40, 40, 40,  # 03-06
        ],
    )

    intervals = build_kp_intervals(df)

    features = build_kp_lag_features(
        intervals,
        [pd.Timestamp("2020-01-01 03:00")],
        lags_hours=(1,),
    )

    # prediction t = 03:00
    # query = t - 1h = 02:00
    # interval [00,03) has not completed by query time 02:00.
    assert np.isnan(features.loc[
        pd.Timestamp("2020-01-01 03:00"),
        "kp_lag_1h",
    ])


def test_kp_lag_1h_becomes_eligible_one_hour_after_interval_end():
    df = make_hourly_kp(
        "2020-01-01 00:00",
        [
            60, 60, 60,
            40, 40, 40,
        ],
    )

    intervals = build_kp_intervals(df)

    features = build_kp_lag_features(
        intervals,
        [pd.Timestamp("2020-01-01 04:00")],
        lags_hours=(1,),
    )

    # t = 04:00
    # query = 03:00
    # [00,03) is now the most recent completed interval.
    assert features.loc[
        pd.Timestamp("2020-01-01 04:00"),
        "kp_lag_1h",
    ] == 6.0


def test_kp_lag_semantics():
    df = make_hourly_kp(
        "2019-12-31 21:00",
        [
            20, 20, 20,  # 21-00
            60, 60, 60,  # 00-03
            40, 40, 40,  # 03-06
        ],
    )

    intervals = build_kp_intervals(df)

    prediction_time = pd.Timestamp("2020-01-01 05:00")

    features = build_kp_lag_features(
        intervals,
        [prediction_time],
        lags_hours=(1, 3),
    )

    # lag 1:
    # q = 04:00
    # latest completed interval = 00-03 => Kp 6
    assert features.loc[
        prediction_time,
        "kp_lag_1h",
    ] == 6.0

    # lag 3:
    # q = 02:00
    # latest completed interval = previous 21-00 => Kp 2
    assert features.loc[
        prediction_time,
        "kp_lag_3h",
    ] == 2.0


def test_kp_midnight_boundary():
    df = make_hourly_kp(
        "2019-12-31 18:00",
        [
            10, 10, 10,  # 18-21
            30, 30, 30,  # 21-00
            50, 50, 50,  # 00-03
        ],
    )

    intervals = build_kp_intervals(df)

    times = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-01-01 01:00",
            "2020-01-01 03:00",
            "2020-01-01 04:00",
        ]
    )

    features = build_kp_lag_features(
        intervals,
        times,
        lags_hours=(1,),
    )

    assert features.loc[
        pd.Timestamp("2020-01-01 00:00"),
        "kp_lag_1h",
    ] == 1.0

    assert features.loc[
        pd.Timestamp("2020-01-01 01:00"),
        "kp_lag_1h",
    ] == 3.0

    assert features.loc[
        pd.Timestamp("2020-01-01 03:00"),
        "kp_lag_1h",
    ] == 3.0

    assert features.loc[
        pd.Timestamp("2020-01-01 04:00"),
        "kp_lag_1h",
    ] == 5.0


def test_kp_year_boundary():
    df = make_hourly_kp(
        "2020-12-31 21:00",
        [
            30, 30, 30,
            60, 60, 60,
        ],
    )

    intervals = build_kp_intervals(df)

    features = build_kp_lag_features(
        intervals,
        [pd.Timestamp("2021-01-01 04:00")],
        lags_hours=(1,),
    )

    assert features.iloc[0]["kp_lag_1h"] == 6.0


def test_kp_leap_year_boundary():
    df = make_hourly_kp(
        "2020-02-28 21:00",
        [
            20, 20, 20,
            50, 50, 50,
        ],
    )

    intervals = build_kp_intervals(df)

    features = build_kp_lag_features(
        intervals,
        [pd.Timestamp("2020-02-29 04:00")],
        lags_hours=(1,),
    )

    assert features.iloc[0]["kp_lag_1h"] == 5.0


def test_kp_future_mutation_invariance():
    original = make_hourly_kp(
        "2020-01-01 00:00",
        [
            20, 20, 20,  # 00-03
            60, 60, 60,  # 03-06
            40, 40, 40,  # 06-09 future
        ],
    )

    mutated = original.copy()

    mutated.loc[
        "2020-01-01 06:00":"2020-01-01 08:00",
        "kp_raw",
    ] = 90

    original_intervals = build_kp_intervals(original)
    mutated_intervals = build_kp_intervals(mutated)

    prediction_time = pd.Timestamp("2020-01-01 05:00")

    original_features = build_kp_lag_features(
        original_intervals,
        [prediction_time],
        lags_hours=(1, 3),
    )

    mutated_features = build_kp_lag_features(
        mutated_intervals,
        [prediction_time],
        lags_hours=(1, 3),
    )

    pd.testing.assert_frame_equal(
        original_features,
        mutated_features,
    )


def test_incomplete_current_interval_cannot_affect_features():
    original = make_hourly_kp(
        "2019-12-31 21:00",
        [
            20, 20, 20,  # previous completed interval
            60, 60, 60,  # 00-03 interval
        ],
    )

    mutated = original.copy()

    mutated.loc[
        "2020-01-01 00:00":"2020-01-01 02:00",
        "kp_raw",
    ] = 90

    original_intervals = build_kp_intervals(original)
    mutated_intervals = build_kp_intervals(mutated)

    prediction_time = pd.Timestamp("2020-01-01 02:00")

    original_features = build_kp_lag_features(
        original_intervals,
        [prediction_time],
        lags_hours=(1,),
    )

    mutated_features = build_kp_lag_features(
        mutated_intervals,
        [prediction_time],
        lags_hours=(1,),
    )

    # q = 01:00. The 00-03 interval has not completed,
    # so changing its retrospective Kp cannot change X(t).
    pd.testing.assert_frame_equal(
        original_features,
        mutated_features,
    )