import numpy as np
import pandas as pd
import pytest
from src.feature_screening.confirmation import (
    PHASE3_ADVANCING_EXPERIMENTS, PHASE3_CONFIRMATION_FOLDS,
    ConfirmationFoldResult, rank_confirmation_candidates,
)

def _r(exp,fold,recall,pr,far,feasible=True):
    return ConfirmationFoldResult(
        exp,fold,0.1 if feasible else None,
        recall if feasible else np.nan,far if feasible else np.nan,
        pr,feasible,pd.DataFrame()
    )

def test_candidates_are_frozen():
    assert PHASE3_ADVANCING_EXPERIMENTS == ("A","E","C")

def test_confirmation_folds_are_frozen():
    assert PHASE3_CONFIRMATION_FOLDS == ("walk_forward_1","walk_forward_2")

def test_final_ranking_uses_worst_fold_recall_first():
    r={}
    for f in PHASE3_CONFIRMATION_FOLDS:
        r[("A",f)] = _r("A",f,0.60 if f=="walk_forward_1" else 0.40,0.3,0.15)
        r[("E",f)] = _r("E",f,0.51 if f=="walk_forward_1" else 0.50,0.3,0.15)
        r[("C",f)] = _r("C",f,0.55 if f=="walk_forward_1" else 0.45,0.3,0.15)
    ranking,selected=rank_confirmation_candidates(r)
    assert selected=="E"
    assert ranking.iloc[0]["minimum_event_recall"]==pytest.approx(0.50)

def test_candidate_infeasible_in_one_fold_cannot_win():
    r={}
    for exp in PHASE3_ADVANCING_EXPERIMENTS:
        for f in PHASE3_CONFIRMATION_FOLDS:
            r[(exp,f)] = _r(exp,f,0.4,0.2,0.1)
    r[("A","walk_forward_2")] = _r("A","walk_forward_2",0,0.9,0,False)
    _,selected=rank_confirmation_candidates(r)
    assert selected=="E"

def test_mean_recall_breaks_equal_worst_fold():
    r={
      ("A","walk_forward_1"):_r("A","walk_forward_1",0.4,0.2,0.1),
      ("A","walk_forward_2"):_r("A","walk_forward_2",0.8,0.2,0.1),
      ("E","walk_forward_1"):_r("E","walk_forward_1",0.4,0.2,0.1),
      ("E","walk_forward_2"):_r("E","walk_forward_2",0.7,0.2,0.1),
      ("C","walk_forward_1"):_r("C","walk_forward_1",0.3,0.9,0.01),
      ("C","walk_forward_2"):_r("C","walk_forward_2",0.9,0.9,0.01),
    }
    _,selected=rank_confirmation_candidates(r)
    assert selected=="A"
