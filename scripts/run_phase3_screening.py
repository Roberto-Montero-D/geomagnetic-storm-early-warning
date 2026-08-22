"""Run the official Phase 3 initial A-E feature screening experiment.

This script performs only the frozen screening fold:
    1996-2016 -> 2017-2018

It does not evaluate Validation 2, Validation 3, or the protected Final Test.
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
from src.dataset.temporal_splits import (
    PERIOD_FINAL_TEST,
    assign_temporal_periods,
)
from src.definitions.events import identify_events
from src.feature_screening.manifests import PHASE3_EXPERIMENT_ORDER
from src.feature_screening.screening import evaluate_phase3_screening


def _progress(message: str) -> None:
    print(message, flush=True)


def build_phase3_screening_inputs(fmt_path: Path, lst_path: Path):
    _progress("[1/6] Loading OMNI source data...")
    omni = load_omni(fmt_path, lst_path)

    _progress("[2/6] Building canonical Kp intervals...")
    kp_intervals = build_kp_intervals(omni)

    _progress("[3/6] Building prediction grid and 93-feature canonical dataset...")
    grid = build_prediction_grid()
    dataset = build_canonical_dataset(omni, kp_intervals, grid)

    _progress("[4/6] Building row status, temporal splits, and protected folds...")
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)
    folds = build_development_folds(dataset, status, splits)

    _progress("[5/6] Identifying canonical storm events...")
    events = identify_events(kp_intervals)

    return dataset, splits, folds, events


def write_phase3_screening_results(result, output_dir: Path) -> None:
    """Write development-only screening artifacts without raw outcome timelines."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result.ranking.to_csv(
        output_dir / "screening_ranking.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "rank": range(1, len(result.advancing_experiments) + 1),
            "experiment": list(result.advancing_experiments),
        }
    ).to_csv(
        output_dir / "screening_advancing_experiments.csv",
        index=False,
    )

    summary_rows = []
    for experiment in PHASE3_EXPERIMENT_ORDER:
        item = result.experiments[experiment]
        summary_rows.append(
            {
                "experiment": experiment,
                "n_features": item.n_features,
                "threshold": item.threshold,
                "event_recall": item.event_recall,
                "false_alarm_rate_per_day": item.false_alarm_rate_per_day,
                "pr_auc": item.pr_auc,
                "operationally_feasible": item.operationally_feasible,
            }
        )
        item.threshold_table.to_csv(
            output_dir / f"screening_{experiment.lower()}_threshold_curve.csv",
            index=False,
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "screening_metrics.csv",
        index=False,
    )


def run_phase3_screening(
    fmt_path: Path,
    lst_path: Path,
    output_dir: Path,
):
    dataset, splits, fold_map, events = build_phase3_screening_inputs(
        fmt_path,
        lst_path,
    )

    if "screening" not in fold_map:
        raise AssertionError("Canonical development folds lack screening fold.")

    if (splits["period"] == PERIOD_FINAL_TEST).sum() == 0:
        raise AssertionError(
            "Canonical split table unexpectedly lacks protected Final Test rows."
        )

    _progress("[6/6] Evaluating frozen Phase 3 A-E screening experiments...")
    result = evaluate_phase3_screening(
        dataset,
        fold_map["screening"],
        events,
        splits,
        progress=True,
    )

    _progress("Writing development-only Phase 3 screening artifacts...")
    write_phase3_screening_results(result, output_dir)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Phase 3 initial feature screening on "
            "1996-2016 -> 2017-2018 only."
        )
    )
    parser.add_argument("--omni-fmt", required=True, type=Path)
    parser.add_argument("--omni-lst", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3/screening"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_phase3_screening(
        args.omni_fmt,
        args.omni_lst,
        args.output_dir,
    )

    print()
    print("Phase 3 initial A-E screening run complete.")
    print(f"Results written to: {args.output_dir}")
    print("Advancing experiments:")
    if result.advancing_experiments:
        for rank, experiment in enumerate(
            result.advancing_experiments,
            start=1,
        ):
            print(f"  {rank}. {experiment}")
    else:
        print("  None — no experiment satisfied the operational constraint.")


if __name__ == "__main__":
    main()
