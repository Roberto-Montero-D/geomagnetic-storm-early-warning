import numpy as np
import pandas as pd
import pytest

from src.features.interactions import (
    INTERACTION_FEATURE_COLUMNS,
    build_interaction_features,
)


def _raw():
    idx = pd.DatetimeIndex(
        ["2020-01-01 12:00", "2020-01-01 13:00"],
        name="prediction_time",
    )
    return pd.DataFrame(
        {
            "bz_gsm": [-5.0, 2.0],
            "bt": [8.0, 7.0],
            "speed": [600.0, 500.0],
            "density": [10.0, 8.0],
            "flow_pressure": [3.0, 2.0],
        },
        index=idx,
    )


def test_interaction_columns_are_frozen():
    result = build_interaction_features(_raw())
    assert tuple(result.columns) == INTERACTION_FEATURE_COLUMNS


def test_bz_negative_part_is_zero_for_positive_bz():
    result = build_interaction_features(_raw())

    assert result.iloc[0]["bz_neg_x_speed"] == 3000.0
    assert result.iloc[0]["bz_neg_x_density"] == 50.0
    assert result.iloc[1]["bz_neg_x_speed"] == 0.0
    assert result.iloc[1]["bz_neg_x_density"] == 0.0


def test_pressure_speed_interaction():
    result = build_interaction_features(_raw())
    assert result.iloc[0]["flow_pressure_x_speed"] == 1800.0


def test_missing_input_propagates_to_affected_interactions():
    raw = _raw()
    raw.loc["2020-01-01 12:00", "speed"] = np.nan

    result = build_interaction_features(raw)

    assert pd.isna(result.iloc[0]["bz_neg_x_speed"])
    assert pd.isna(result.iloc[0]["flow_pressure_x_speed"])
    assert result.iloc[0]["bz_neg_x_density"] == 50.0


def test_missing_required_column_raises():
    with pytest.raises(KeyError):
        build_interaction_features(_raw().drop(columns=["density"]))
