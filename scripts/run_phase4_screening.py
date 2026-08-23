"""Run the official Phase 4 initial imbalance screening.

This script evaluates only the frozen screening fold:
    Initial Train -> Validation 1

It does not evaluate Validation 2, Validation 3, or protected Final Test.
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
from src.imbalance.contract import PHASE4_EXPERIMENT_NAMES
from src.imbalance.screening import (
    evaluate_imbalance_experiment,
    rank_imbalance_experiments,
    Phase4ScreeningResult,
)


def _progress(message: str) -> None:
    print(message, flush=True)


def build_phase4_screening_inputs(fmt_path: Path, lst_path: Path):
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


def evaluate_phase4_screening_with_progress(dataset,fold,events,splits):
    results={}
    total=len(PHASE4_EXPERIMENT_NAMES)
    for i,name in enumerate(PHASE4_EXPERIMENT_NAMES,start=1):
        _progress(f"    [{i:02d}/{total:02d}] {name}")
        results[name]=evaluate_imbalance_experiment(
            dataset,fold,events,splits,name
        )
    ranking,advancing=rank_imbalance_experiments(results)
    return Phase4ScreeningResult(results,ranking,advancing)


def write_phase4_screening_results(result,output_dir: Path) -> None:
    """Write development-only aggregate artifacts, not raw outcome timelines."""
    output_dir.mkdir(parents=True,exist_ok=True)

    result.ranking.to_csv(output_dir/"screening_ranking.csv",index=False)

    pd.DataFrame({
        "rank":range(1,len(result.advancing_experiments)+1),
        "experiment":list(result.advancing_experiments),
    }).to_csv(output_dir/"screening_advancing_experiments.csv",index=False)

    rows=[]
    for name in PHASE4_EXPERIMENT_NAMES:
        item=result.experiments[name]
        rows.append({
            "experiment":name,
            "threshold":item.threshold,
            "event_recall":item.event_recall,
            "false_alarm_rate_per_day":item.false_alarm_rate_per_day,
            "pr_auc":item.pr_auc,
            "operationally_feasible":item.operationally_feasible,
        })
        item.threshold_table.to_csv(
            output_dir/f"screening_{name}_threshold_curve.csv",
            index=False,
        )

    pd.DataFrame(rows).to_csv(output_dir/"screening_metrics.csv",index=False)


def run_phase4_screening(fmt_path: Path,lst_path: Path,output_dir: Path):
    dataset,splits,fold_map,events=build_phase4_screening_inputs(fmt_path,lst_path)

    if "screening" not in fold_map:
        raise AssertionError("Canonical development folds lack screening fold.")

    if (splits["period"]==PERIOD_FINAL_TEST).sum()==0:
        raise AssertionError(
            "Canonical split table unexpectedly lacks protected Final Test rows."
        )

    _progress("[6/6] Evaluating frozen Phase 4 imbalance configurations...")
    result=evaluate_phase4_screening_with_progress(
        dataset,fold_map["screening"],events,splits
    )

    _progress("Writing development-only Phase 4 screening artifacts...")
    write_phase4_screening_results(result,output_dir)
    return result


def parse_args():
    parser=argparse.ArgumentParser(
        description="Run frozen Phase 4 imbalance screening on Validation 1 only."
    )
    parser.add_argument("--omni-fmt",required=True,type=Path)
    parser.add_argument("--omni-lst",required=True,type=Path)
    parser.add_argument(
        "--output-dir",type=Path,default=Path("results/phase4/screening")
    )
    return parser.parse_args()


def main():
    args=parse_args()
    result=run_phase4_screening(args.omni_fmt,args.omni_lst,args.output_dir)

    print()
    print("Phase 4 initial imbalance screening run complete.")
    print(f"Results written to: {args.output_dir}")
    print("Advancing configurations:")
    if result.advancing_experiments:
        for rank,name in enumerate(result.advancing_experiments,start=1):
            print(f"  {rank}. {name}")
    else:
        print("  None — no configuration satisfied the operational constraint.")


if __name__=="__main__":
    main()
