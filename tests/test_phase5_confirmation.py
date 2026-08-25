import numpy as np
import pandas as pd
from src.model_selection.confirmation import (
    ConfirmationFoldResult,PHASE5_CONFIRMATION_CANDIDATES,
    PHASE5_CONFIRMATION_FOLDS,rank_confirmation_candidates,
)

def _r(f,c,recall,pr,far,ok=True):
    return ConfirmationFoldResult(f,c,.2 if ok else None,recall,far,pr,ok,pd.DataFrame())

def _results():
    out={}
    for f in PHASE5_CONFIRMATION_FOLDS:
        for c in PHASE5_CONFIRMATION_CANDIDATES:
            out[(f,c)]=_r(f,c,.5,.4,.18)
    return out

def test_worst_fold_recall_is_primary_confirmation_metric():
    x=_results(); a,b=PHASE5_CONFIRMATION_CANDIDATES[:2]
    x[("walk_forward_1",a)]=_r("walk_forward_1",a,.9,.2,.18)
    x[("walk_forward_2",a)]=_r("walk_forward_2",a,.4,.2,.18)
    x[("walk_forward_1",b)]=_r("walk_forward_1",b,.6,.2,.18)
    x[("walk_forward_2",b)]=_r("walk_forward_2",b,.5,.2,.18)
    assert rank_confirmation_candidates(x).iloc[0].config_id==b

def test_candidate_must_be_feasible_in_both_folds():
    x=_results(); a=PHASE5_CONFIRMATION_CANDIDATES[0]
    x[("walk_forward_2",a)]=_r("walk_forward_2",a,np.nan,.9,np.nan,False)
    assert rank_confirmation_candidates(x).iloc[-1].config_id==a

def test_mean_recall_breaks_worst_fold_tie():
    x=_results(); a,b=PHASE5_CONFIRMATION_CANDIDATES[:2]
    x[("walk_forward_1",a)]=_r("walk_forward_1",a,.5,.2,.18)
    x[("walk_forward_2",a)]=_r("walk_forward_2",a,.7,.2,.18)
    x[("walk_forward_1",b)]=_r("walk_forward_1",b,.5,.9,.10)
    x[("walk_forward_2",b)]=_r("walk_forward_2",b,.6,.9,.10)
    assert rank_confirmation_candidates(x).iloc[0].config_id==a

def test_mean_pr_auc_breaks_recall_ties():
    x=_results(); a,b=PHASE5_CONFIRMATION_CANDIDATES[:2]
    for f in PHASE5_CONFIRMATION_FOLDS:
        x[(f,a)]=_r(f,a,.6,.4,.18)
        x[(f,b)]=_r(f,b,.6,.5,.19)
    assert rank_confirmation_candidates(x).iloc[0].config_id==b

def test_lower_mean_far_breaks_remaining_tie():
    x=_results(); a,b=PHASE5_CONFIRMATION_CANDIDATES[:2]
    for f in PHASE5_CONFIRMATION_FOLDS:
        x[(f,a)]=_r(f,a,.6,.5,.18)
        x[(f,b)]=_r(f,b,.6,.5,.10)
    assert rank_confirmation_candidates(x).iloc[0].config_id==b
