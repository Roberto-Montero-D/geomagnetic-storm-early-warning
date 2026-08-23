import numpy as np
import pandas as pd
import pytest
from src.imbalance.contract import PHASE4_EXPERIMENT_NAMES
from src.imbalance.screening import (
    ImbalanceScreeningResult, rank_imbalance_experiments
)

def _result(name,recall,pr,far,feasible=True):
    return ImbalanceScreeningResult(
        name,0.5 if feasible else None,recall,far,pr,feasible,
        pd.Series(dtype=float),pd.DataFrame()
    )

def test_ranking_uses_frozen_operational_order():
    results={}
    for name in PHASE4_EXPERIMENT_NAMES:
        results[name]=_result(name,0.1,0.1,0.19)
    results["smote_k3"]=_result("smote_k3",0.8,0.2,0.19)
    results["class_weight_10"]=_result("class_weight_10",0.7,0.5,0.18)
    results["undersample_5_to_1"]=_result("undersample_5_to_1",0.7,0.4,0.10)
    ranking,advancing=rank_imbalance_experiments(results)
    assert list(ranking.experiment[:3]) == [
        "smote_k3","class_weight_10","undersample_5_to_1"
    ]
    assert advancing == tuple(ranking.experiment[:3])

def test_infeasible_never_advances_over_feasible():
    results={n:_result(n,0.1,0.1,0.19) for n in PHASE4_EXPERIMENT_NAMES}
    results["smote_k3"]=_result("smote_k3",1.0,1.0,np.nan,False)
    ranking,advancing=rank_imbalance_experiments(results)
    assert "smote_k3" not in advancing
    assert ranking.iloc[-1].experiment=="smote_k3"

def test_frozen_order_breaks_exact_ties():
    results={n:_result(n,0.5,0.2,0.1) for n in PHASE4_EXPERIMENT_NAMES}
    ranking,advancing=rank_imbalance_experiments(results)
    assert tuple(ranking.experiment)==PHASE4_EXPERIMENT_NAMES
    assert advancing==PHASE4_EXPERIMENT_NAMES[:3]

def test_exactly_top_three_feasible_advance():
    results={n:_result(n,0.5,0.2,0.1) for n in PHASE4_EXPERIMENT_NAMES}
    _,advancing=rank_imbalance_experiments(results)
    assert len(advancing)==3

def test_fewer_than_three_feasible_advances_only_feasible():
    results={n:_result(n,np.nan,0.1,np.nan,False) for n in PHASE4_EXPERIMENT_NAMES}
    results["none"]=_result("none",0.4,0.2,0.19)
    results["class_weight_1"]=_result("class_weight_1",0.3,0.3,0.18)
    _,advancing=rank_imbalance_experiments(results)
    assert advancing==("none","class_weight_1")
