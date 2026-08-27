import pandas as pd
from src.final_test.diagnostics_physical import (
 FROZEN_FEATURES,event_feature_snapshots,detected_vs_missed_summary,
 false_alarm_feature_snapshots,
)
def _dataset():
 idx=pd.date_range("2022-01-01",periods=12,freq="h")
 return pd.DataFrame({f:range(12) for f in FROZEN_FEATURES},index=idx)
def test_frozen_features_are_phase3_a():
 assert len(FROZEN_FEATURES)==10
def test_event_window_excludes_event_start():
 events=pd.DataFrame({"event_id":[1],"start_time":[pd.Timestamp("2022-01-01 06:00")],"detected":[True],"year":[2022]})
 out=event_feature_snapshots(_dataset(),events)
 assert out.iloc[0]["window_rows"]==6
 # values 0..5; event-start value 6 must not enter
 assert out.iloc[0][f"{FROZEN_FEATURES[0]}__max"]==5
def test_comparison_preserves_detected_missed_groups():
 s=pd.DataFrame({"detected":[True,False],**{f"{f}__median":[1.,2.] for f in FROZEN_FEATURES}})
 out=detected_vs_missed_summary(s)
 assert set(out["group"])=={"detected","missed"}
 assert len(out)==20
def test_false_alarm_snapshots_only_use_false_alarms():
 eps=pd.DataFrame({"first_alert_time":[pd.Timestamp("2022-01-01 01:00"),pd.Timestamp("2022-01-01 02:00")],
 "classification":["false_alarm","early_detection"]})
 out=false_alarm_feature_snapshots(_dataset(),eps)
 assert len(out)==1
 assert out.iloc[0]["first_alert_time"]==pd.Timestamp("2022-01-01 01:00")
