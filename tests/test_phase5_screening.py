import numpy as np
import pandas as pd
import pytest
from src.model_selection.contract import PHASE5_CONFIGURATIONS
from src.model_selection.screening import ModelScreeningResult,advance_family_winners,rank_family_configurations

def _r(c,f,recall=.5,pr=.2,far=.18,ok=True):
    return ModelScreeningResult(c,f,.5 if ok else None,recall,far,pr,ok,pd.Series(dtype=float),pd.DataFrame())

def _all():
    return {c.config_id:_r(c.config_id,c.family) for c in PHASE5_CONFIGURATIONS}

@pytest.mark.parametrize("family",["extratrees","lightgbm","xgboost"])
def test_frozen_tie_order(family):
    ranking=rank_family_configurations(_all(),family)
    expected=tuple(c.config_id for c in PHASE5_CONFIGURATIONS if c.family==family)
    assert tuple(ranking.config_id)==expected

def test_recall_precedes_pr_auc():
    x=_all(); et=[c for c in PHASE5_CONFIGURATIONS if c.family=="extratrees"]
    x[et[0].config_id]=_r(et[0].config_id,"extratrees",.7,.1,.19)
    x[et[1].config_id]=_r(et[1].config_id,"extratrees",.6,.9,.10)
    assert rank_family_configurations(x,"extratrees").iloc[0].config_id==et[0].config_id

def test_pr_auc_breaks_recall_tie():
    x=_all(); cs=[c for c in PHASE5_CONFIGURATIONS if c.family=="lightgbm"]
    x[cs[0].config_id]=_r(cs[0].config_id,"lightgbm",.6,.4,.18)
    x[cs[1].config_id]=_r(cs[1].config_id,"lightgbm",.6,.5,.19)
    assert rank_family_configurations(x,"lightgbm").iloc[0].config_id==cs[1].config_id

def test_lower_far_breaks_remaining_tie():
    x=_all(); cs=[c for c in PHASE5_CONFIGURATIONS if c.family=="xgboost"]
    x[cs[0].config_id]=_r(cs[0].config_id,"xgboost",.6,.5,.19)
    x[cs[1].config_id]=_r(cs[1].config_id,"xgboost",.6,.5,.10)
    assert rank_family_configurations(x,"xgboost").iloc[0].config_id==cs[1].config_id

def test_infeasible_cannot_win():
    x=_all(); et=[c for c in PHASE5_CONFIGURATIONS if c.family=="extratrees"]
    x[et[0].config_id]=_r(et[0].config_id,"extratrees",1,1,np.nan,False)
    assert rank_family_configurations(x,"extratrees").iloc[-1].config_id==et[0].config_id

def test_one_winner_per_family_advances():
    rankings,adv=advance_family_winners(_all())
    assert set(rankings)=={"extratrees","lightgbm","xgboost"}
    assert len(adv)==3

def test_no_feasible_family_does_not_advance():
    x=_all()
    for c in PHASE5_CONFIGURATIONS:
        if c.family=="lightgbm": x[c.config_id]=_r(c.config_id,c.family,np.nan,.2,np.nan,False)
    _,adv=advance_family_winners(x)
    assert len(adv)==2 and not any(a.startswith("lightgbm_") for a in adv)

def test_missing_result_rejected():
    x=_all(); cid=next(c.config_id for c in PHASE5_CONFIGURATIONS if c.family=="xgboost"); del x[cid]
    with pytest.raises(ValueError,match="Missing Phase 5 xgboost"):
        rank_family_configurations(x,"xgboost")
