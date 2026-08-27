"""Run Phase 9.2 physical error-regime diagnostics."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scripts.run_phase5_screening import build_phase5_screening_inputs
from src.final_test.diagnostics_physical import (
    event_feature_snapshots, detected_vs_missed_summary,
    false_alarm_feature_snapshots, yearly_false_alarm_physics,
)

def run_phase9_2(omni_fmt,omni_lst,phase8_dir,phase9_1_dir,output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    dataset,_splits,_folds,_events=build_phase5_screening_inputs(omni_fmt,omni_lst)
    outcomes=pd.read_csv(Path(phase9_1_dir)/"event_outcomes.csv")
    episodes=pd.read_csv(Path(phase8_dir)/"final_test_alert_episodes.csv")
    snapshots=event_feature_snapshots(dataset,outcomes)
    comparison=detected_vs_missed_summary(snapshots)
    false_snapshots=false_alarm_feature_snapshots(dataset,episodes)
    false_yearly=yearly_false_alarm_physics(false_snapshots)
    snapshots.to_csv(output_dir/"event_feature_snapshots.csv",index=False)
    comparison.to_csv(output_dir/"detected_vs_missed_physics.csv",index=False)
    false_snapshots.to_csv(output_dir/"false_alarm_feature_snapshots.csv",index=False)
    false_yearly.to_csv(output_dir/"yearly_false_alarm_physics.csv",index=False)
    print("Phase 9.2 physical error-regime diagnostics complete.")
    print("No model fitting performed.")
    print("No threshold search performed.")
    print("No feature selection performed.")
    print(f"Event snapshots: {len(snapshots)}")
    print(f"Detected: {int(snapshots['detected'].sum())}")
    print(f"Missed: {int((~snapshots['detected']).sum())}")
    print(f"False-alarm snapshots: {len(false_snapshots)}")
    print(f"Frozen diagnostic features: {len(comparison['feature'].unique())}")
    print(f"Results written to: {output_dir}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--omni-fmt",type=Path,required=True); p.add_argument("--omni-lst",type=Path,required=True)
    p.add_argument("--phase8-dir",type=Path,default=Path("results/phase8/final_test"))
    p.add_argument("--phase9-1-dir",type=Path,default=Path("results/phase9/operational_temporal"))
    p.add_argument("--output-dir",type=Path,default=Path("results/phase9/physical_error_regimes"))
    a=p.parse_args(); run_phase9_2(a.omni_fmt,a.omni_lst,a.phase8_dir,a.phase9_1_dir,a.output_dir)
if __name__=="__main__": main()
