"""Phase 9.2 post-hoc physical error-regime diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.feature_screening.manifests import PHASE3_FEATURE_SETS

FROZEN_FEATURES = tuple(PHASE3_FEATURE_SETS["A"])
PRE_EVENT_HOURS = 6

def _validate_dataset(dataset):
    missing=set(FROZEN_FEATURES)-set(dataset.columns)
    if missing: raise ValueError(f"dataset missing frozen Phase 3 A features: {sorted(missing)}")
    if not isinstance(dataset.index,pd.DatetimeIndex): raise TypeError("dataset index must be DatetimeIndex.")
    return dataset

def event_feature_snapshots(dataset, event_outcomes, *, pre_event_hours=PRE_EVENT_HOURS):
    """Summarize only causally available frozen features in [start-H, start)."""
    dataset=_validate_dataset(dataset)
    required={"event_id","start_time","detected","year"}
    missing=required-set(event_outcomes.columns)
    if missing: raise ValueError(f"event outcomes missing columns: {sorted(missing)}")
    outcomes=event_outcomes.copy()
    outcomes["start_time"]=pd.to_datetime(outcomes["start_time"],errors="raise")
    rows=[]
    for event in outcomes.itertuples(index=False):
        lo=event.start_time-pd.Timedelta(hours=pre_event_hours)
        window=dataset.loc[(dataset.index>=lo)&(dataset.index<event.start_time),FROZEN_FEATURES]
        row={"event_id":event.event_id,"year":int(event.year),"detected":bool(event.detected),
             "window_rows":int(len(window))}
        for feature in FROZEN_FEATURES:
            values=pd.to_numeric(window[feature],errors="coerce")
            row[f"{feature}__median"]=float(values.median()) if values.notna().any() else np.nan
            row[f"{feature}__min"]=float(values.min()) if values.notna().any() else np.nan
            row[f"{feature}__max"]=float(values.max()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def detected_vs_missed_summary(snapshots):
    """Descriptive median/IQR comparison; no tests, ranking, or selection."""
    rows=[]
    for feature in FROZEN_FEATURES:
        col=f"{feature}__median"
        for label,detected in (("detected",True),("missed",False)):
            x=pd.to_numeric(snapshots.loc[snapshots["detected"]==detected,col],errors="coerce").dropna()
            rows.append({"feature":feature,"group":label,"n":int(len(x)),
                         "median":np.nan if x.empty else float(x.median()),
                         "p25":np.nan if x.empty else float(x.quantile(.25)),
                         "p75":np.nan if x.empty else float(x.quantile(.75))})
    return pd.DataFrame(rows)

def false_alarm_feature_snapshots(dataset, episodes):
    """Feature state at immutable false-alarm episode first-alert timestamps."""
    dataset=_validate_dataset(dataset)
    required={"first_alert_time","classification"}
    missing=required-set(episodes.columns)
    if missing: raise ValueError(f"episodes missing columns: {sorted(missing)}")
    eps=episodes.copy()
    eps["first_alert_time"]=pd.to_datetime(eps["first_alert_time"],errors="raise")
    eps=eps.loc[eps["classification"]=="false_alarm"].copy()
    rows=[]
    for ep in eps.itertuples(index=False):
        t=ep.first_alert_time
        row={"first_alert_time":t,"year":int(t.year)}
        if t in dataset.index:
            for feature in FROZEN_FEATURES:
                row[feature]=dataset.at[t,feature]
        else:
            for feature in FROZEN_FEATURES: row[feature]=np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def yearly_false_alarm_physics(false_alarm_snapshots):
    rows=[]
    for year,g in false_alarm_snapshots.groupby("year"):
        for feature in FROZEN_FEATURES:
            x=pd.to_numeric(g[feature],errors="coerce").dropna()
            rows.append({"year":int(year),"feature":feature,"n":int(len(x)),
                         "median":np.nan if x.empty else float(x.median()),
                         "p25":np.nan if x.empty else float(x.quantile(.25)),
                         "p75":np.nan if x.empty else float(x.quantile(.75))})
    return pd.DataFrame(rows)
