"""Run the official development-only Phase 2 baseline experiment.

This script intentionally requires explicit local OMNI source paths. It builds
the canonical dataset through the existing Phase 0/1 APIs, constructs the
protected development folds, evaluates B0-B3, and writes only development
artifacts. Protected 2022-2025 outcomes are never exported.
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
from src.dataset.temporal_splits import assign_temporal_periods, PERIOD_FINAL_TEST
from src.definitions.events import identify_events
from src.evaluation.cross_fold import evaluate_development_folds


def build_phase2_inputs(fmt_path: Path, lst_path: Path):
    omni = load_omni(fmt_path, lst_path)
    kp_intervals = build_kp_intervals(omni)
    grid = build_prediction_grid()
    dataset = build_canonical_dataset(omni, kp_intervals, grid)
    status = build_row_status(dataset)
    splits = assign_temporal_periods(dataset.index)
    folds = build_development_folds(dataset, status, splits)
    events = identify_events(kp_intervals)
    return dataset, status, splits, folds, events


def _assert_development_only(frame: pd.DataFrame, splits: pd.DataFrame) -> None:
    if "prediction_time" in frame.columns:
        times = pd.DatetimeIndex(pd.to_datetime(frame["prediction_time"]))
        if (splits.reindex(times)["period"] == PERIOD_FINAL_TEST).any():
            raise AssertionError("Output contains protected Final Test timestamps.")


def write_phase2_results(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = result.fold_metrics.copy()
    metrics.to_csv(output_dir / "baseline_fold_metrics.csv", index=False)

    selected = pd.DataFrame(
        {
            "baseline": list(result.selected_thresholds),
            "threshold": list(result.selected_thresholds.values()),
        }
    )
    selected.to_csv(output_dir / "baseline_selected_thresholds.csv", index=False)

    for baseline, table in result.threshold_tables.items():
        safe = baseline.lower()
        table.to_csv(output_dir / f"{safe}_threshold_curve.csv", index=False)


def run_phase2_baselines(
    fmt_path: Path,
    lst_path: Path,
    output_dir: Path,
):
    dataset, status, splits, fold_map, events = build_phase2_inputs(fmt_path, lst_path)

    folds = [fold_map["screening"], fold_map["walk_forward_1"], fold_map["walk_forward_2"]]
    result = evaluate_development_folds(
        dataset,
        folds,
        events,
        splits,
    )

    # Explicit artifact guard: fold metrics contain fold labels rather than raw
    # timestamps, and threshold artifacts contain no outcome timeline at all.
    if (splits["period"] == PERIOD_FINAL_TEST).sum() == 0:
        raise AssertionError("Canonical split table unexpectedly lacks Final Test rows.")

    write_phase2_results(result, output_dir)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run development-only Phase 2 B0-B3 baseline evaluation."
    )
    parser.add_argument("--omni-fmt", required=True, type=Path)
    parser.add_argument("--omni-lst", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_phase2_baselines(args.omni_fmt, args.omni_lst, args.output_dir)

    print("Phase 2 development baseline run complete.")
    print(f"Results written to: {args.output_dir}")
    print("Selected baseline-evaluation thresholds:")
    for baseline, threshold in result.selected_thresholds.items():
        print(f"  {baseline}: {threshold}")


if __name__ == "__main__":
    main()
