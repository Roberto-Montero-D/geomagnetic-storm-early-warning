import numpy as np
import pandas as pd
from types import SimpleNamespace
from src.imbalance.confirmation import (
    PHASE4_ADVANCING_EXPERIMENTS, ConfirmationFoldResult,
    rank_confirmation_candidates,
)

def _r(fold,name,recall,pr,far,ok=True):
    return ConfirmationFoldResult(fold,name,0.2 if ok else None,
        recall,far,pr,ok,pd.DataFrame())

def test_advancing_set_is_exactly_frozen_screening_top_three():
    assert PHASE4_ADVANCING_EXPERIMENTS == (
        "undersample_10_to_1","none","class_weight_1"
    )

def test_confirmation_ranking_prioritizes_worst_fold_recall():
    results={}
    vals={
        "undersample_10_to_1":((0.60,0.40),(0.30,0.40)),
        "none":((0.50,0.30),(0.50,0.30)),
        "class_weight_1":((0.45,0.60),(0.45,0.60)),
    }
    for name,pairs in vals.items():
        for fold,(recall,pr) in zip(("walk_forward_1","walk_forward_2"),pairs):
            results[(fold,name)] = _r(fold,name,recall,pr,0.18)
    ranking,selected=rank_confirmation_candidates(results)
    assert selected=="none"
    assert ranking.iloc[0].minimum_event_recall==0.5

def test_mean_recall_breaks_equal_worst_fold():
    results={}
    data={
        "undersample_10_to_1":(0.5,0.8),
        "none":(0.5,0.6),
        "class_weight_1":(0.4,0.9),
    }
    for name,(a,b) in data.items():
        results[("walk_forward_1",name)]=_r("walk_forward_1",name,a,0.2,0.18)
        results[("walk_forward_2",name)]=_r("walk_forward_2",name,b,0.2,0.18)
    _,selected=rank_confirmation_candidates(results)
    assert selected=="undersample_10_to_1"

def test_infeasible_candidate_cannot_win():
    results={}
    for name in PHASE4_ADVANCING_EXPERIMENTS:
        results[("walk_forward_1",name)]=_r("walk_forward_1",name,0.4,0.2,0.18)
        results[("walk_forward_2",name)]=_r("walk_forward_2",name,0.4,0.2,0.18)
    results[("walk_forward_2","undersample_10_to_1")] = _r(
        "walk_forward_2","undersample_10_to_1",np.nan,0.9,np.nan,False)
    _,selected=rank_confirmation_candidates(results)
    assert selected=="none"

def test_frozen_order_breaks_exact_tie():
    results={}
    for name in PHASE4_ADVANCING_EXPERIMENTS:
        for fold in ("walk_forward_1","walk_forward_2"):
            results[(fold,name)]=_r(fold,name,0.5,0.3,0.18)
    ranking,selected=rank_confirmation_candidates(results)
    assert tuple(ranking.experiment)==PHASE4_ADVANCING_EXPERIMENTS
    assert selected=="undersample_10_to_1"
