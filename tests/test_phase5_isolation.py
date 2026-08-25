import pandas as pd
import pytest

from src.baselines.framework import DevelopmentFold
from src.dataset.temporal_splits import assign_temporal_periods
from src.model_selection.contract import PHASE5_CONFIGURATIONS
from src.model_selection.isolation import (
    assert_identical_phase5_screening_indices,
    validate_phase5_screening_fold,
)


def _base():
    train=pd.date_range("2016-12-31 20:00",periods=4,freq="h",name="prediction_time")
    val=pd.date_range("2017-01-01 00:00",periods=4,freq="h",name="prediction_time")
    index=train.append(val)
    dataset=pd.DataFrame({"target":[0,1,0,1,0,1,0,1]},index=index)
    splits=assign_temporal_periods(index)
    fold=DevelopmentFold("screening",train,val)
    return dataset,splits,fold


def test_exact_screening_contract_passes():
    dataset,splits,fold=_base()
    validate_phase5_screening_fold(dataset,fold,splits)


def test_non_screening_fold_name_rejected():
    dataset,splits,fold=_base()
    bad=DevelopmentFold("walk_forward_1",fold.train_index,fold.validation_index)
    with pytest.raises(ValueError,match="frozen `screening` fold"):
        validate_phase5_screening_fold(dataset,bad,splits)


def test_later_validation_period_rejected():
    dataset,splits,fold=_base()
    later=pd.date_range("2019-01-01",periods=4,freq="h",name="prediction_time")
    extra=dataset.iloc[:4].copy(); extra.index=later
    augmented=pd.concat([dataset,extra]).sort_index()
    new_splits=assign_temporal_periods(augmented.index)
    bad=DevelopmentFold("screening",fold.train_index,later)
    with pytest.raises(ValueError,match="Validation 1.*2017-2018"):
        validate_phase5_screening_fold(augmented,bad,new_splits)


def test_final_test_validation_rejected_explicitly():
    dataset,splits,fold=_base()
    final=pd.date_range("2022-01-01",periods=4,freq="h",name="prediction_time")
    extra=dataset.iloc[:4].copy(); extra.index=final
    augmented=pd.concat([dataset,extra]).sort_index()
    new_splits=assign_temporal_periods(augmented.index)
    bad=DevelopmentFold("screening",fold.train_index,final)
    with pytest.raises(ValueError,match="protected Final Test"):
        validate_phase5_screening_fold(augmented,bad,new_splits)


def test_validation1_cannot_enter_training_rows():
    dataset,splits,fold=_base()
    contaminated=fold.train_index.append(fold.validation_index[:1])
    bad=DevelopmentFold("screening",contaminated,fold.validation_index[1:])
    with pytest.raises(ValueError,match="only Initial Train"):
        validate_phase5_screening_fold(dataset,bad,splits)


def test_train_validation_overlap_rejected():
    dataset,splits,fold=_base()
    bad=DevelopmentFold(
        "screening",
        fold.train_index.append(fold.validation_index[:1]),
        fold.validation_index,
    )
    with pytest.raises(ValueError,match="must not overlap"):
        validate_phase5_screening_fold(dataset,bad,splits)


def test_split_alignment_must_be_exact():
    dataset,splits,fold=_base()
    bad=splits.iloc[::-1]
    with pytest.raises(ValueError,match="align exactly"):
        validate_phase5_screening_fold(dataset,fold,bad)


def test_all_27_identical_indices_pass():
    _,_,fold=_base()
    observed={
        c.config_id:(fold.train_index.copy(),fold.validation_index.copy())
        for c in PHASE5_CONFIGURATIONS
    }
    assert_identical_phase5_screening_indices(observed,fold)


def test_single_configuration_index_drift_is_detected():
    _,_,fold=_base()
    observed={
        c.config_id:(fold.train_index.copy(),fold.validation_index.copy())
        for c in PHASE5_CONFIGURATIONS
    }
    first=PHASE5_CONFIGURATIONS[0].config_id
    observed[first]=(fold.train_index[:-1],fold.validation_index)
    with pytest.raises(AssertionError,match="canonical screening training"):
        assert_identical_phase5_screening_indices(observed,fold)


def test_missing_configuration_in_index_audit_is_detected():
    _,_,fold=_base()
    observed={
        c.config_id:(fold.train_index.copy(),fold.validation_index.copy())
        for c in PHASE5_CONFIGURATIONS
    }
    del observed[PHASE5_CONFIGURATIONS[-1].config_id]
    with pytest.raises(AssertionError,match="index audit mismatch"):
        assert_identical_phase5_screening_indices(observed,fold)
