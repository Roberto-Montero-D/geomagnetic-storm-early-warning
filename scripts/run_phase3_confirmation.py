"""Run frozen Phase 3 walk-forward confirmation for A/E/C."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.baselines.framework import build_development_folds
from src.data.kp import build_kp_intervals
from src.data.omni import load_omni
from src.dataset.builder import build_canonical_dataset
from src.dataset.prediction_grid import build_prediction_grid
from src.dataset.row_status import build_row_status
from src.dataset.temporal_splits import PERIOD_FINAL_TEST, assign_temporal_periods
from src.definitions.events import identify_events
from src.feature_screening.confirmation import (
    PHASE3_ADVANCING_EXPERIMENTS, PHASE3_CONFIRMATION_FOLDS,
    evaluate_phase3_confirmation,
)

def _progress(message): print(message,flush=True)

def build_phase3_confirmation_inputs(fmt_path,lst_path):
    _progress("[1/6] Loading OMNI source data...")
    omni=load_omni(fmt_path,lst_path)
    _progress("[2/6] Building canonical Kp intervals...")
    kp=build_kp_intervals(omni)
    _progress("[3/6] Building prediction grid and canonical dataset...")
    dataset=build_canonical_dataset(omni,kp,build_prediction_grid())
    _progress("[4/6] Building row status, temporal splits, and development folds...")
    status=build_row_status(dataset); splits=assign_temporal_periods(dataset.index)
    folds=build_development_folds(dataset,status,splits)
    _progress("[5/6] Identifying canonical storm events...")
    events=identify_events(kp)
    return dataset,splits,folds,events

def write_phase3_confirmation_results(result,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    result.ranking.to_csv(output_dir/"confirmation_ranking.csv",index=False)
    rows=[]
    for fold in PHASE3_CONFIRMATION_FOLDS:
        for exp in PHASE3_ADVANCING_EXPERIMENTS:
            item=result.fold_results[(exp,fold)]
            rows.append({"fold":fold,"experiment":exp,"threshold":item.threshold,
                "event_recall":item.event_recall,
                "false_alarm_rate_per_day":item.false_alarm_rate_per_day,
                "pr_auc":item.pr_auc,
                "operationally_feasible":item.operationally_feasible})
            item.threshold_table.to_csv(
                output_dir/f"{fold}_{exp.lower()}_threshold_curve.csv",index=False)
    pd.DataFrame(rows).to_csv(output_dir/"confirmation_fold_metrics.csv",index=False)
    pd.DataFrame([{"selected_experiment":result.selected_experiment}]).to_csv(
        output_dir/"confirmation_selected_experiment.csv",index=False)

def run_phase3_confirmation(fmt_path,lst_path,output_dir):
    dataset,splits,folds,events=build_phase3_confirmation_inputs(fmt_path,lst_path)
    missing=[f for f in PHASE3_CONFIRMATION_FOLDS if f not in folds]
    if missing: raise AssertionError(f"Canonical development folds missing: {missing}")
    if (splits["period"]==PERIOD_FINAL_TEST).sum()==0:
        raise AssertionError("Canonical split table unexpectedly lacks protected Final Test rows.")
    _progress("[6/6] Evaluating frozen A/E/C walk-forward confirmation...")
    result=evaluate_phase3_confirmation(dataset,folds,events,splits,progress=True)
    _progress("Writing development-only Phase 3 confirmation artifacts...")
    write_phase3_confirmation_results(result,output_dir)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--omni-fmt",required=True,type=Path)
    p.add_argument("--omni-lst",required=True,type=Path)
    p.add_argument("--output-dir",type=Path,default=Path("results/phase3/confirmation"))
    a=p.parse_args()
    r=run_phase3_confirmation(a.omni_fmt,a.omni_lst,a.output_dir)
    print("\nPhase 3 walk-forward confirmation complete.")
    print(f"Results written to: {a.output_dir}")
    print(f"Selected experiment: {r.selected_experiment}")
if __name__=="__main__": main()
