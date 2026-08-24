"""Run official Phase 4 walk-forward confirmation.

Only the three strategies frozen by Phase 4 screening are evaluated, on:
    WF1 -> Validation 2
    WF2 -> Validation 3

Protected Final Test is never evaluated.
"""
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
from src.imbalance.confirmation import (
    PHASE4_ADVANCING_EXPERIMENTS,
    evaluate_confirmation_fold,
    rank_confirmation_candidates,
    Phase4ConfirmationResult,
)
from src.imbalance.contract import PHASE4_CONFIRMATION_FOLDS


def _progress(message: str) -> None:
    print(message,flush=True)


def build_phase4_confirmation_inputs(fmt_path: Path,lst_path: Path):
    _progress("[1/6] Loading OMNI source data...")
    omni=load_omni(fmt_path,lst_path)

    _progress("[2/6] Building canonical Kp intervals...")
    kp_intervals=build_kp_intervals(omni)

    _progress("[3/6] Building canonical prediction grid and dataset...")
    grid=build_prediction_grid()
    dataset=build_canonical_dataset(omni,kp_intervals,grid)

    _progress("[4/6] Building row status, temporal splits, and development folds...")
    status=build_row_status(dataset)
    splits=assign_temporal_periods(dataset.index)
    folds=build_development_folds(dataset,status,splits)

    _progress("[5/6] Identifying canonical storm events...")
    events=identify_events(kp_intervals)
    return dataset,splits,folds,events


def evaluate_confirmation_with_progress(dataset,folds,events,splits):
    results={}
    total=len(PHASE4_CONFIRMATION_FOLDS)*len(PHASE4_ADVANCING_EXPERIMENTS)
    i=0
    for fold_name in PHASE4_CONFIRMATION_FOLDS:
        if fold_name not in folds:
            raise AssertionError(f"Missing canonical confirmation fold: {fold_name}")
        for experiment in PHASE4_ADVANCING_EXPERIMENTS:
            i += 1
            _progress(f"    [{i:02d}/{total:02d}] {fold_name} / {experiment}")
            results[(fold_name,experiment)]=evaluate_confirmation_fold(
                dataset,folds[fold_name],events,splits,fold_name,experiment
            )

    ranking,selected=rank_confirmation_candidates(results)
    return Phase4ConfirmationResult(results,ranking,selected)


def write_phase4_confirmation_results(result,output_dir: Path) -> None:
    output_dir.mkdir(parents=True,exist_ok=True)

    rows=[]
    for fold_name in PHASE4_CONFIRMATION_FOLDS:
        for experiment in PHASE4_ADVANCING_EXPERIMENTS:
            item=result.folds[(fold_name,experiment)]
            rows.append({
                "fold":fold_name,
                "experiment":experiment,
                "threshold":item.threshold,
                "event_recall":item.event_recall,
                "false_alarm_rate_per_day":item.false_alarm_rate_per_day,
                "pr_auc":item.pr_auc,
                "operationally_feasible":item.operationally_feasible,
            })
            item.threshold_table.to_csv(
                output_dir/f"{fold_name}_{experiment}_threshold_curve.csv",
                index=False,
            )

    pd.DataFrame(rows).to_csv(
        output_dir/"confirmation_fold_metrics.csv",index=False
    )
    result.ranking.to_csv(
        output_dir/"confirmation_ranking.csv",index=False
    )
    pd.DataFrame({
        "selected_experiment":[result.selected_experiment]
    }).to_csv(
        output_dir/"confirmation_selected_experiment.csv",index=False
    )


def run_phase4_confirmation(fmt_path: Path,lst_path: Path,output_dir: Path):
    dataset,splits,folds,events=build_phase4_confirmation_inputs(fmt_path,lst_path)

    if (splits["period"]==PERIOD_FINAL_TEST).sum()==0:
        raise AssertionError(
            "Canonical split table unexpectedly lacks protected Final Test rows."
        )

    _progress("[6/6] Evaluating frozen Phase 4 confirmation candidates...")
    result=evaluate_confirmation_with_progress(dataset,folds,events,splits)

    _progress("Writing development-only Phase 4 confirmation artifacts...")
    write_phase4_confirmation_results(result,output_dir)
    return result


def parse_args():
    parser=argparse.ArgumentParser(
        description="Run frozen Phase 4 WF1/WF2 imbalance confirmation."
    )
    parser.add_argument("--omni-fmt",required=True,type=Path)
    parser.add_argument("--omni-lst",required=True,type=Path)
    parser.add_argument(
        "--output-dir",type=Path,default=Path("results/phase4/confirmation")
    )
    return parser.parse_args()


def main():
    args=parse_args()
    result=run_phase4_confirmation(
        args.omni_fmt,args.omni_lst,args.output_dir
    )

    print()
    print("Phase 4 walk-forward confirmation run complete.")
    print(f"Results written to: {args.output_dir}")
    print(f"Selected imbalance strategy: {result.selected_experiment}")


if __name__=="__main__":
    main()
