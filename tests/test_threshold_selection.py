import numpy as np
import pandas as pd
import pytest

from src.dataset.temporal_splits import assign_temporal_periods
from src.evaluation.threshold_selection import (
    DEFAULT_MAX_FAR_PER_DAY,
    DEFAULT_THRESHOLD_GRID,
    select_operational_threshold,
)

def _events():
    return pd.DataFrame({
        "event_id": [1],
        "start_time": [pd.Timestamp("2021-01-01 10:00")],
        "end_time": [pd.Timestamp("2021-01-01 12:00")],
        "boundary_status": ["complete"],
    })

def _probabilities():
    idx = pd.date_range("2021-01-01", periods=24, freq="h", name="prediction_time")
    p = pd.Series(0.0, index=idx, name="probability")
    p.loc["2021-01-01 02:00"] = 0.4
    p.loc["2021-01-01 08:00"] = 0.8
    return p

def test_frozen_default_grid_and_far_limit():
    assert DEFAULT_THRESHOLD_GRID[0] == 0.01
    assert DEFAULT_THRESHOLD_GRID[-1] == 0.99
    assert len(DEFAULT_THRESHOLD_GRID) == 99
    assert DEFAULT_MAX_FAR_PER_DAY == 0.2

def test_selects_minimum_feasible_threshold():
    p = _probabilities()
    splits = assign_temporal_periods(p.index)
    result = select_operational_threshold(
        p, _events(), splits,
        thresholds=[0.3, 0.5, 0.9],
        max_far_per_day=0.2,
    )
    assert result.selected_threshold == 0.5
    assert result.table["far_feasible"].tolist() == [False, True, True]

def test_no_feasible_threshold_returns_none():
    p = _probabilities()
    p.loc["2021-01-01 00:00"] = 1.0
    p.loc["2021-01-01 20:00"] = 1.0
    splits = assign_temporal_periods(p.index)
    result = select_operational_threshold(
        p, _events(), splits,
        thresholds=[0.1, 0.5, 0.9],
        max_far_per_day=0.0,
    )
    assert result.selected_threshold is None

def test_final_test_predictions_are_rejected():
    idx = pd.DatetimeIndex(["2022-01-01 00:00"], name="prediction_time")
    p = pd.Series([0.1], index=idx, name="probability")
    splits = assign_temporal_periods(idx)
    empty_events = pd.DataFrame(columns=["event_id","start_time","end_time","boundary_status"])
    with pytest.raises(ValueError, match="Final Test predictions"):
        select_operational_threshold(p, empty_events, splits, thresholds=[0.5])

def test_final_test_events_are_rejected():
    p = _probabilities()
    splits = assign_temporal_periods(p.index)
    events = pd.DataFrame({
        "event_id":[1],
        "start_time":[pd.Timestamp("2022-01-01")],
        "end_time":[pd.Timestamp("2022-01-01 03:00")],
        "boundary_status":["complete"],
    })
    with pytest.raises(ValueError, match="Final Test events"):
        select_operational_threshold(p, events, splits, thresholds=[0.5])

def test_unsorted_or_duplicate_thresholds_raise():
    p = _probabilities()
    splits = assign_temporal_periods(p.index)
    with pytest.raises(ValueError, match="sorted"):
        select_operational_threshold(p, _events(), splits, thresholds=[0.5,0.3])
    with pytest.raises(ValueError, match="unique"):
        select_operational_threshold(p, _events(), splits, thresholds=[0.5,0.5])
