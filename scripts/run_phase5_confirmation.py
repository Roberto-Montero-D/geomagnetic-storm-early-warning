"""Run Phase 5 walk-forward confirmation."""
from __future__ import annotations
import argparse
from pathlib import Path
from time import perf_counter
import pandas as pd

from scripts.run_phase5_screening import build_phase5_screening_inputs, _progress
from src.model_selection.confirmation import (
    PHASE5_CONFIRMATION_CANDIDATES,PHASE5_CONFIRMATION_FOLDS,
    Phase5ConfirmationResult,evaluate_confirmation_fold,rank_confirmation_candidates,
)


def evaluate_confirmation_with_progress(dataset,folds,events,splits):
    results={}
    total=len(PHASE5_CONFIRMATION_CANDIDATES)*len(PHASE5_CONFIRMATION_FOLDS)
    i=0
    for fold_name in PHASE5_CONFIRMATION_FOLDS:
        for cid in PHASE5_CONFIRMATION_CANDIDATES:
            i+=1
            start=perf_counter()
            _progress(f"    [{i:02d}/{total:02d}] {fold_name} / {cid}")
            results[(fold_name,cid)]=evaluate_confirmation_fold(
                dataset,folds[fold_name],events,splits,fold_name,cid)
            _progress(f"             completed in {perf_counter()-start:.1f} s")
    ranking=rank_confirmation_candidates(results)
    feasible=ranking[ranking.feasible_both_folds]
    selected=None if feasible.empty else str(feasible.iloc[0].config_id)
    return Phase5ConfirmationResult(results,ranking,selected)


def write_confirmation_results(result,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for fold_name in PHASE5_CONFIRMATION_FOLDS:
        for cid in PHASE5_CONFIRMATION_CANDIDATES:
            r=result.fold_results[(fold_name,cid)]
            rows.append({
                "fold_name":fold_name,"config_id":cid,"threshold":r.threshold,
                "event_recall":r.event_recall,
                "false_alarm_rate_per_day":r.false_alarm_rate_per_day,
                "pr_auc":r.pr_auc,"operationally_feasible":r.operationally_feasible,
            })
            r.threshold_table.to_csv(
                output_dir/f"confirmation_{fold_name}_{cid}_threshold_curve.csv",
                index=False)
    pd.DataFrame(rows).to_csv(output_dir/"confirmation_fold_metrics.csv",index=False)
    result.ranking.to_csv(output_dir/"confirmation_ranking.csv",index=False)
    pd.DataFrame([{"selected_config_id":result.selected_config_id}]).to_csv(
        output_dir/"confirmation_selected_model.csv",index=False)


def run_phase5_confirmation(fmt,lst,output_dir):
    dataset,splits,folds,events=build_phase5_screening_inputs(fmt,lst)
    _progress("[7/7] Evaluating 6 frozen Phase 5 walk-forward confirmation fits...")
    result=evaluate_confirmation_with_progress(dataset,folds,events,splits)
    _progress("Writing development-only Phase 5 confirmation artifacts...")
    write_confirmation_results(result,output_dir)
    return result


def parse_args():
    p=argparse.ArgumentParser(description="Run frozen Phase 5 WF1/WF2 confirmation.")
    p.add_argument("--omni-fmt",required=True,type=Path)
    p.add_argument("--omni-lst",required=True,type=Path)
    p.add_argument("--output-dir",type=Path,default=Path("results/phase5/confirmation"))
    return p.parse_args()


def main():
    a=parse_args()
    r=run_phase5_confirmation(a.omni_fmt,a.omni_lst,a.output_dir)
    print()
    print("Phase 5 walk-forward confirmation complete.")
    print(f"Results written to: {a.output_dir}")
    print(f"Selected configuration: {r.selected_config_id}")


if __name__=="__main__":
    main()
