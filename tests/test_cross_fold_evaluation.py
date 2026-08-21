import numpy as np
import pandas as pd
from src.baselines.framework import DevelopmentFold
from src.evaluation.cross_fold import (
    DETERMINISTIC_THRESHOLD,
    evaluate_development_folds,
)
from src.features.raw import PRIMARY_RAW_FEATURE_COLUMNS

def _dataset_and_folds():
    idx1=pd.date_range("2017-01-01",periods=30,freq="h",name="prediction_time")
    idx2=pd.date_range("2019-01-01",periods=30,freq="h",name="prediction_time")
    idx3=pd.date_range("2021-01-01",periods=30,freq="h",name="prediction_time")
    index=idx1.append(idx2).append(idx3)
    rng=np.random.default_rng(9)
    d=pd.DataFrame(rng.normal(size=(90,len(PRIMARY_RAW_FEATURE_COLUMNS))),
                   index=index,columns=list(PRIMARY_RAW_FEATURE_COLUMNS))
    d["bz_gsm"]=rng.normal(0,5,90); d["bt"]=np.abs(rng.normal(7,2,90))
    d["speed"]=rng.normal(500,60,90); d["density"]=np.abs(rng.normal(6,2,90))
    d["flow_pressure"]=np.abs(rng.normal(2,.5,90))
    for c in ["kp_lag_1h","kp_lag_3h","kp_lag_6h","kp_lag_12h","kp_lag_24h"]:
        d[c]=rng.uniform(0,8,90)
    d["target"]=[0,1]*45
    folds=[
        DevelopmentFold("f1",idx1[:20],idx1[20:]),
        DevelopmentFold("f2",idx1.append(idx2[:20]),idx2[20:]),
        DevelopmentFold("f3",idx1.append(idx2).append(idx3[:20]),idx3[20:]),
    ]
    return d,folds

def _events():
    return pd.DataFrame({
        "event_id":[1,2,3],
        "start_time":[pd.Timestamp("2017-01-02 04:00"),pd.Timestamp("2019-01-02 04:00"),pd.Timestamp("2021-01-02 04:00")],
        "end_time":[pd.Timestamp("2017-01-02 05:00"),pd.Timestamp("2019-01-02 05:00"),pd.Timestamp("2021-01-02 05:00")],
        "boundary_status":["complete"]*3,
    })

def test_cross_fold_returns_four_by_three_metric_rows():
    d,folds=_dataset_and_folds()
    r=evaluate_development_folds(d,folds,_events(),thresholds=[0.25,0.5,0.75])
    assert len(r.fold_metrics)==12
    assert set(r.fold_metrics["baseline"])=={"B0_persistence","B1_physical","B2_logistic","B3_extratrees"}
    assert set(r.fold_metrics["fold"])=={"f1","f2","f3"}

def test_deterministic_threshold_is_fixed_not_selected():
    d,folds=_dataset_and_folds()
    r=evaluate_development_folds(d,folds,_events(),thresholds=[0.25,0.5,0.75])
    assert r.selected_thresholds["B0_persistence"]==DETERMINISTIC_THRESHOLD
    assert r.selected_thresholds["B1_physical"]==DETERMINISTIC_THRESHOLD
    assert "B0_persistence" not in r.threshold_tables
    assert "B1_physical" not in r.threshold_tables

def test_probabilistic_threshold_tables_preserve_requested_grid():
    d,folds=_dataset_and_folds()
    r=evaluate_development_folds(d,folds,_events(),thresholds=[0.25,0.5,0.75])
    for name in ("B2_logistic","B3_extratrees"):
        assert r.threshold_tables[name]["threshold"].tolist()==[0.25,0.5,0.75]

def test_fold_metrics_keep_validation_windows_separate():
    d,folds=_dataset_and_folds()
    r=evaluate_development_folds(d,folds,_events(),thresholds=[0.5])
    assert r.fold_metrics.groupby(["baseline","fold"]).size().eq(1).all()
