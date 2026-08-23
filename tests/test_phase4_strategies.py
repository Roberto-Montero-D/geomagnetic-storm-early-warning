import numpy as np
import pandas as pd
import pytest

from src.imbalance.contract import PHASE4_EXPERIMENTS
from src.imbalance.strategies import prepare_training_data


def _experiment(name):
    return next(x for x in PHASE4_EXPERIMENTS if x.name==name)


def _data():
    rng=np.random.default_rng(123)
    # Enough minority samples for every frozen k-neighbor configuration.
    y=pd.Series([0]*180+[1]*12,index=pd.RangeIndex(192),name="target")
    x=pd.DataFrame(
        rng.normal(size=(192,3)),
        index=y.index,
        columns=["a","b","c"],
    )
    # Make minority structure non-degenerate.
    x.loc[y.eq(1),"a"] += np.linspace(0.0,2.0,12)
    return x,y


def test_none_returns_unchanged_copies():
    x,y=_data()
    out=prepare_training_data(x,y,_experiment("none"))
    pd.testing.assert_frame_equal(out.x,x)
    pd.testing.assert_series_equal(out.y,y)
    assert out.class_weight is None
    assert out.x is not x and out.y is not y


@pytest.mark.parametrize("weight",[1,3,5,10,20,50])
def test_class_weight_contract(weight):
    x,y=_data()
    out=prepare_training_data(x,y,_experiment(f"class_weight_{weight}"))
    pd.testing.assert_frame_equal(out.x,x)
    pd.testing.assert_series_equal(out.y,y)
    assert out.class_weight == {0:1.0,1:float(weight)}


@pytest.mark.parametrize(
    ("name","negative_per_positive"),
    [("undersample_10_to_1",10),("undersample_5_to_1",5),("undersample_2_to_1",2)],
)
def test_undersampling_exact_ratio(name,negative_per_positive):
    x,y=_data()
    out=prepare_training_data(x,y,_experiment(name))
    counts=out.y.value_counts()
    assert counts[1] == 12
    assert counts[0] == negative_per_positive*counts[1]
    assert out.class_weight is None


@pytest.mark.parametrize("name",["smote_k3","smote_k5","smote_k7"])
def test_smote_balances_classes(name):
    x,y=_data()
    out=prepare_training_data(x,y,_experiment(name))
    counts=out.y.value_counts()
    assert counts[0] == counts[1] == 180


@pytest.mark.parametrize(
    "name",["borderline_smote_k3","borderline_smote_k5","borderline_smote_k7"]
)
def test_borderline_smote_never_changes_original_inputs(name):
    x,y=_data()
    x_before=x.copy(deep=True); y_before=y.copy(deep=True)
    prepare_training_data(x,y,_experiment(name))
    pd.testing.assert_frame_equal(x,x_before)
    pd.testing.assert_series_equal(y,y_before)


def test_smote_enn_is_deterministic():
    x,y=_data()
    a=prepare_training_data(x,y,_experiment("smote_enn"))
    b=prepare_training_data(x,y,_experiment("smote_enn"))
    pd.testing.assert_frame_equal(a.x,b.x)
    pd.testing.assert_series_equal(a.y,b.y)


@pytest.mark.parametrize(
    "name",
    ["undersample_5_to_1","smote_k5","borderline_smote_k5","smote_enn"],
)
def test_resampling_is_deterministic(name):
    x,y=_data()
    a=prepare_training_data(x,y,_experiment(name))
    b=prepare_training_data(x,y,_experiment(name))
    pd.testing.assert_frame_equal(a.x,b.x)
    pd.testing.assert_series_equal(a.y,b.y)


def test_api_cannot_receive_validation_data():
    x,y=_data()
    validation=x.iloc[:5].copy()
    with pytest.raises(TypeError):
        prepare_training_data(x,y,_experiment("smote_k3"),validation)


def test_original_training_inputs_are_never_mutated():
    x,y=_data()
    x0=x.copy(deep=True); y0=y.copy(deep=True)
    for experiment in PHASE4_EXPERIMENTS:
        prepare_training_data(x,y,experiment)
        pd.testing.assert_frame_equal(x,x0)
        pd.testing.assert_series_equal(y,y0)


def test_rejects_misaligned_training_indices():
    x,y=_data()
    bad=y.copy()
    bad.index=pd.RangeIndex(1,len(bad)+1)
    with pytest.raises(ValueError,match="indices"):
        prepare_training_data(x,bad,_experiment("none"))


def test_rejects_nonbinary_target():
    x,y=_data()
    bad=y.copy(); bad.iloc[0]=2
    with pytest.raises(ValueError,match="binary"):
        prepare_training_data(x,bad,_experiment("none"))
