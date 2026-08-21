import numpy as np
import pandas as pd
import pytest
from src.evaluation.operational import binary_predictions_as_probabilities, evaluate_operational_series

def _events():
    return pd.DataFrame({
        "event_id":[1],
        "start_time":[pd.Timestamp("2021-01-01 10:00")],
        "end_time":[pd.Timestamp("2021-01-01 12:00")],
        "boundary_status":["complete"],
    })

def test_binary_adapter_preserves_missing_and_index():
    idx=pd.date_range("2021-01-01",periods=3,freq="h")
    x=pd.Series([0,1,pd.NA],index=idx,dtype="Int8")
    p=binary_predictions_as_probabilities(x)
    assert p.tolist()[:2]==[0.0,1.0]
    assert np.isnan(p.iloc[2])
    pd.testing.assert_index_equal(p.index,idx)

def test_binary_adapter_rejects_nonbinary_values():
    with pytest.raises(ValueError,match="only 0, 1"):
        binary_predictions_as_probabilities(pd.Series([0,2,1]))

def test_operational_evaluation_reuses_episode_semantics():
    idx=pd.date_range("2021-01-01 00:00",periods=24,freq="h")
    p=pd.Series(0.0,index=idx,name="probability")
    p.loc["2021-01-01 05:00"]=0.9   # false alarm
    p.loc["2021-01-01 06:00"]=0.0
    p.loc["2021-01-01 08:00"]=0.9   # gap 3h => same episode as 05:00
    episodes,m=evaluate_operational_series(p,_events(),threshold=0.5)
    assert len(episodes)==1
    assert episodes.iloc[0]["classification"]=="early_detection"
    assert m.event_recall==1.0
    assert m.false_alarm_rate_per_day==0.0
    assert m.median_lead_time==pd.Timedelta(hours=5)

def test_false_alarm_rate_uses_valid_exposure():
    idx=pd.date_range("2021-01-01 00:00",periods=24,freq="h")
    p=pd.Series(0.0,index=idx,name="probability")
    p.loc["2021-01-01 00:00"]=0.9
    _,m=evaluate_operational_series(p,_events(),threshold=0.5)
    assert m.false_alarm_rate_per_day==1.0

def test_missing_probability_reduces_exposure_not_negative():
    idx=pd.date_range("2021-01-01 00:00",periods=24,freq="h")
    p=pd.Series(np.nan,index=idx,name="probability")
    p.iloc[:12]=0.0
    p.iloc[0]=0.9
    _,m=evaluate_operational_series(p,_events(),threshold=0.5)
    assert m.false_alarm_rate_per_day==2.0

def test_late_detection_counts_for_event_recall_but_not_lead_time():
    idx=pd.date_range("2021-01-01 00:00",periods=24,freq="h")
    p=pd.Series(0.0,index=idx,name="probability")
    p.loc["2021-01-01 11:00"]=0.9
    _,m=evaluate_operational_series(p,_events(),threshold=0.5)
    assert m.event_recall==1.0
    assert m.n_late_detections==1
    assert pd.isna(m.median_lead_time)

def test_no_alerts_gives_zero_recall_and_zero_far():
    idx=pd.date_range("2021-01-01 00:00",periods=24,freq="h")
    p=pd.Series(0.0,index=idx,name="probability")
    episodes,m=evaluate_operational_series(p,_events(),threshold=0.5)
    assert episodes.empty
    assert m.event_recall==0.0
    assert m.false_alarm_rate_per_day==0.0
