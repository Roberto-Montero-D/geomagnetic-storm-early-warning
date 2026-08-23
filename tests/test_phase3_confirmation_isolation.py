import pandas as pd
import pytest

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import assign_temporal_periods
from src.feature_screening.confirmation import (
    PHASE3_CONFIRMATION_PERIODS,
    _validate_confirmation_fold,
)


def _indices():
    initial = pd.date_range(
        "2016-12-29", periods=72, freq="h", name="prediction_time"
    )
    val1 = pd.date_range(
        "2017-01-01", periods=48, freq="h", name="prediction_time"
    )
    val2 = pd.date_range(
        "2019-01-01", periods=48, freq="h", name="prediction_time"
    )
    val3 = pd.date_range(
        "2021-01-01", periods=48, freq="h", name="prediction_time"
    )
    return initial, val1, val2, val3


def _splits(*indices):
    index = indices[0]
    for other in indices[1:]:
        index = index.append(other)
    return assign_temporal_periods(index)


def test_confirmation_period_contract_is_frozen():
    assert PHASE3_CONFIRMATION_PERIODS["walk_forward_1"]["validation"] == (
        "validation_2",
    )
    assert PHASE3_CONFIRMATION_PERIODS["walk_forward_2"]["validation"] == (
        "validation_3",
    )


def test_valid_walk_forward_1_contract_passes():
    initial, val1, val2, val3 = _indices()
    fold = DevelopmentFold("walk_forward_1", initial.append(val1), val2)
    _validate_confirmation_fold(
        fold, _splits(initial, val1, val2, val3)
    )


def test_valid_walk_forward_2_contract_passes():
    initial, val1, val2, val3 = _indices()
    fold = DevelopmentFold(
        "walk_forward_2", initial.append(val1).append(val2), val3
    )
    _validate_confirmation_fold(
        fold, _splits(initial, val1, val2, val3)
    )


def test_walk_forward_1_rejects_validation_3_row():
    initial, val1, val2, val3 = _indices()
    fold = DevelopmentFold(
        "walk_forward_1",
        initial.append(val1),
        val2.append(val3[:1]),
    )
    with pytest.raises(ValueError, match="frozen atomic periods"):
        _validate_confirmation_fold(
            fold, _splits(initial, val1, val2, val3)
        )


def test_walk_forward_2_rejects_validation_3_in_training():
    initial, val1, val2, val3 = _indices()
    fold = DevelopmentFold(
        "walk_forward_2",
        initial.append(val1).append(val2).append(val3[:1]),
        val3[1:],
    )
    with pytest.raises(ValueError, match="frozen atomic periods"):
        _validate_confirmation_fold(
            fold, _splits(initial, val1, val2, val3)
        )


def test_final_test_row_is_rejected_explicitly():
    initial, val1, val2, val3 = _indices()
    final = pd.date_range(
        "2022-01-01", periods=2, freq="h", name="prediction_time"
    )
    fold = DevelopmentFold(
        "walk_forward_1",
        initial.append(val1),
        val2.append(final),
    )
    with pytest.raises(ValueError, match="protected Final Test"):
        _validate_confirmation_fold(
            fold, _splits(initial, val1, val2, val3, final)
        )


def test_timestamp_absent_from_split_table_is_rejected():
    initial, val1, val2, val3 = _indices()
    unknown = pd.DatetimeIndex(
        [pd.Timestamp("2019-02-01")], name="prediction_time"
    )
    fold = DevelopmentFold(
        "walk_forward_1", initial.append(val1), val2.append(unknown)
    )
    with pytest.raises(ValueError, match="absent from splits"):
        _validate_confirmation_fold(
            fold, _splits(initial, val1, val2, val3)
        )
