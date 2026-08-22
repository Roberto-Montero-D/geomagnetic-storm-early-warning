"""Phase 2 baseline runner with observational progress reporting."""
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
from src.dataset.temporal_splits import assign_temporal_periods, PERIOD_FINAL_TEST
from src.definitions.events import identify_events
from src.evaluation.cross_fold import evaluate_development_folds

def _progress(message: str) -> None:
    print(message, flush=True)

def build_phase2_inputs(fmt_path: Path, lst_path: Path):
    _progress("[1/6] Loading OMNI source data...")
    omni=load_omni(fmt_path,lst_path)
    _progress("[2/6] Building canonical Kp intervals...")
    kp_intervals=build_kp_intervals(omni)
    _progress("[3/6] Building prediction grid and 93-feature canonical dataset...")
    grid=build_prediction_grid()
    dataset=build_canonical_dataset(omni,kp_intervals,grid)
    _progress("[4/6] Building row status, temporal splits, and protected folds...")
    status=build_row_status(dataset)
    splits=assign_temporal_periods(dataset.index)
    folds=build_development_folds(dataset,status,splits)
    _progress("[5/6] Identifying canonical storm events...")
    events=identify_events(kp_intervals)
    return dataset,status,splits,folds,events

def write_phase2_results(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True,exist_ok=True)
    result.fold_metrics.to_csv(output_dir/"baseline_fold_metrics.csv",index=False)
    pd.DataFrame({"baseline":list(result.selected_thresholds),
                  "threshold":list(result.selected_thresholds.values())}).to_csv(
        output_dir/"baseline_selected_thresholds.csv",index=False)
    for baseline,table in result.threshold_tables.items():
        table.to_csv(output_dir/f"{baseline.lower()}_threshold_curve.csv",index=False)

def run_phase2_baselines(fmt_path: Path,lst_path: Path,output_dir: Path):
    dataset,status,splits,fold_map,events=build_phase2_inputs(fmt_path,lst_path)
    folds=[fold_map["screening"],fold_map["walk_forward_1"],fold_map["walk_forward_2"]]
    _progress("[6/6] Evaluating B0-B3 on protected development folds...")
    result=evaluate_development_folds(dataset,folds,events,splits,progress=True)
    if (splits["period"]==PERIOD_FINAL_TEST).sum()==0:
        raise AssertionError("Canonical split table unexpectedly lacks Final Test rows.")
    _progress("Writing development-only Phase 2 result artifacts...")
    write_phase2_results(result,output_dir)
    return result

def parse_args():
    p=argparse.ArgumentParser(description="Run development-only Phase 2 B0-B3 baseline evaluation.")
    p.add_argument("--omni-fmt",required=True,type=Path)
    p.add_argument("--omni-lst",required=True,type=Path)
    p.add_argument("--output-dir",type=Path,default=Path("results/phase2"))
    return p.parse_args()

def main():
    a=parse_args()
    result=run_phase2_baselines(a.omni_fmt,a.omni_lst,a.output_dir)
    print("\nPhase 2 development baseline run complete.")
    print(f"Results written to: {a.output_dir}")
    print("Selected baseline-evaluation thresholds:")
    for baseline,threshold in result.selected_thresholds.items():
        print(f"  {baseline}: {threshold}")

if __name__=="__main__":
    main()
