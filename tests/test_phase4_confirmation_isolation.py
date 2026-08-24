import numpy as np
import pandas as pd
import pytest

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import assign_temporal_periods
from src.imbalance.confirmation import _validate_confirmation_fold


def _canonical_indices():
    # Small slices from every development atomic period. assign_temporal_periods
    # supplies the canonical labels used by the validator.
    initial = pd.date_range(
        "2016-12-30 00:00", periods=48, freq="h", name="prediction_time"
    )
    val1 = pd.date_range(
        "2017-01-01 00:00", periods=48, freq="h", name="prediction_time"
    )
    val2 = pd.date_range(
        "2019-01-01 00:00", periods=48, freq="h", name="prediction_time"
    )
    val3 = pd.date_range(
        "2021-01-01 00:00", periods=48, freq="h", name="prediction_time"
    )
    final = pd.date_range(
        "2022-01-01 00:00", periods=4, freq="h", name="prediction_time"
    )
    index = initial.append(val1).append(val2).append(val3).append(final)
    return initial, val1, val2, val3, final, assign_temporal_periods(index)


def test_wf1_exact_contract_passes():
    initial,val1,val2,_,_,splits=_canonical_indices()
    fold=DevelopmentFold("walk_forward_1",initial.append(val1),val2)
    _validate_confirmation_fold(fold,splits,"walk_forward_1")


def test_wf2_exact_contract_passes():
    initial,val1,val2,val3,_,splits=_canonical_indices()
    fold=DevelopmentFold(
        "walk_forward_2",initial.append(val1).append(val2),val3
    )
    _validate_confirmation_fold(fold,splits,"walk_forward_2")


def test_wf1_rejects_partial_validation_period():
    initial,val1,val2,_,_,splits=_canonical_indices()
    fold=DevelopmentFold("walk_forward_1",initial.append(val1),val2[:-1])
    with pytest.raises(ValueError,match="validation rows"):
        _validate_confirmation_fold(fold,splits,"walk_forward_1")


def test_wf2_rejects_partial_training_period():
    initial,val1,val2,val3,_,splits=_canonical_indices()
    fold=DevelopmentFold(
        "walk_forward_2",
        initial.append(val1).append(val2[:-1]),
        val3,
    )
    with pytest.raises(ValueError,match="training rows"):
        _validate_confirmation_fold(fold,splits,"walk_forward_2")


def test_confirmation_rejects_final_test_row():
    initial,val1,val2,_,final,splits=_canonical_indices()
    bad=DevelopmentFold(
        "walk_forward_1",
        initial.append(val1),
        val2.append(final[:1]),
    )
    with pytest.raises(ValueError,match="protected Final Test"):
        _validate_confirmation_fold(bad,splits,"walk_forward_1")


def test_confirmation_rejects_timestamp_absent_from_splits():
    initial,val1,val2,_,_,splits=_canonical_indices()
    unknown=pd.DatetimeIndex(
        [pd.Timestamp("2019-06-01 00:00")],name="prediction_time"
    )
    bad=DevelopmentFold(
        "walk_forward_1",
        initial.append(val1),
        val2.append(unknown),
    )
    with pytest.raises(ValueError,match="absent from splits"):
        _validate_confirmation_fold(bad,splits,"walk_forward_1")
