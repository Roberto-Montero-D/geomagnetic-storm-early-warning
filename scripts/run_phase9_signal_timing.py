"""Run Phase 9.5 onset-centered signal-timing diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.run_phase5_screening import build_phase5_screening_inputs
from src.final_test.diagnostics_recurrence import event_context_table
from src.final_test.diagnostics_signal_timing import (
    grouped_timing_contrast_summary,
    grouped_trajectory_summary,
    onset_aligned_event_trajectory,
    timing_window_contrast,
    trajectory_coverage_summary,
)


def run_phase9_5(
    omni_fmt: Path,
    omni_lst: Path,
    phase8_dir: Path,
    phase9_1_dir: Path,
    output_dir: Path,
) -> None:
    phase8_dir = Path(phase8_dir)
    phase9_1_dir = Path(phase9_1_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Checking immutable dependencies...")
    prediction_path = (
        phase8_dir / "final_test_predictions.csv"
    )
    outcomes_path = (
        phase9_1_dir / "event_outcomes.csv"
    )
    for path in (prediction_path, outcomes_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )
    print("      complete")

    print("[2/7] Rebuilding canonical causal dataset and events...")
    dataset, _splits, _folds, events = (
        build_phase5_screening_inputs(
            omni_fmt,
            omni_lst,
        )
    )
    print(
        "      complete: "
        f"{len(dataset):,} prediction rows; "
        f"{len(events):,} canonical events"
    )

    print("[3/7] Loading protected event outcomes...")
    outcomes = pd.read_csv(outcomes_path)
    print(f"      complete: {len(outcomes):,} event rows")

    print("[4/7] Reconstructing protected event context...")
    context = event_context_table(
        events,
        outcomes,
        dataset,
    )
    print(f"      complete: {len(context):,} event rows")

    print("[5/7] Loading immutable Phase 8 probabilities...")
    predictions = pd.read_csv(
        prediction_path,
        parse_dates=["prediction_time"],
    ).set_index("prediction_time")
    print(
        "      complete: "
        f"{len(predictions):,} protected prediction rows"
    )

    print("[6/7] Building onset-centered signal trajectories...")
    trajectory = onset_aligned_event_trajectory(
        context,
        dataset,
        predictions,
    )
    summary = grouped_trajectory_summary(trajectory)
    contrasts = timing_window_contrast(trajectory)
    contrast_summary = grouped_timing_contrast_summary(
        contrasts
    )
    coverage = trajectory_coverage_summary(trajectory)
    print(
        "      complete: "
        f"{len(trajectory):,} aligned event-hour rows"
    )

    print("[7/7] Writing diagnostic outputs...")
    trajectory.to_csv(
        output_dir / "onset_aligned_event_trajectory.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "grouped_trajectory_summary.csv",
        index=False,
    )
    contrasts.to_csv(
        output_dir / "event_timing_window_contrasts.csv",
        index=False,
    )
    contrast_summary.to_csv(
        output_dir / "grouped_timing_contrast_summary.csv",
        index=False,
    )
    coverage.to_csv(
        output_dir / "trajectory_coverage.csv",
        index=False,
    )
    print("      complete")

    print()
    print("Phase 9.5 signal-timing diagnostics complete.")
    print("No model fitting performed.")
    print("No new probabilities generated.")
    print("No threshold search performed.")
    print("No feature selection performed.")
    print("Official Phase 8 result unchanged.")
    print()

    print(
        "Detected vs missed: median change from "
        "-12..-7 h to -6..-1 h"
    )
    view = contrast_summary.loc[
        contrast_summary["grouping"] == "outcome",
        [
            "outcome_group",
            "metric",
            "n_events",
            "median_warning_minus_early",
        ],
    ]
    print(view.to_string(index=False))

    print()
    print(
        "2022-2024 vs 2025 by outcome: "
        "median timing contrasts"
    )
    view2 = contrast_summary.loc[
        contrast_summary["grouping"] == "outcome_period",
        [
            "outcome_group",
            "comparison_period",
            "metric",
            "n_events",
            "median_warning_minus_early",
        ],
    ]
    print(view2.to_string(index=False))

    print()
    print(f"Results written to: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run Phase 9.5 post-hoc onset-centered "
            "signal-timing diagnostics."
        )
    )
    p.add_argument(
        "--omni-fmt",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--omni-lst",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--phase8-dir",
        type=Path,
        default=Path("results/phase8/final_test"),
    )
    p.add_argument(
        "--phase9-1-dir",
        type=Path,
        default=Path(
            "results/phase9/operational_temporal"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/phase9/signal_timing"
        ),
    )
    return p.parse_args()


def main():
    a = parse_args()
    run_phase9_5(
        a.omni_fmt,
        a.omni_lst,
        a.phase8_dir,
        a.phase9_1_dir,
        a.output_dir,
    )


if __name__ == "__main__":
    main()
