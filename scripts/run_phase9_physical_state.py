"""Run Phase 9.4 pre-onset physical-state diagnostics.

Post-hoc descriptive analysis only. This script does not fit models,
generate probabilities, search thresholds, or select features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.run_phase5_screening import build_phase5_screening_inputs
from src.final_test.diagnostics_physical_state import (
    conditional_2025_feature_shift,
    conditional_recall_summary,
    detected_vs_missed_physical_summary,
    pre_onset_physical_state_table,
)
from src.final_test.diagnostics_recurrence import event_context_table


def run_phase9_4(
    omni_fmt: Path,
    omni_lst: Path,
    phase9_1_dir: Path,
    output_dir: Path,
) -> None:
    phase9_1_dir = Path(phase9_1_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Checking Phase 9.1 dependencies...")
    outcomes_path = phase9_1_dir / "event_outcomes.csv"
    if not outcomes_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 9.1 event outcomes: {outcomes_path}"
        )
    print("      complete")

    print("[2/6] Rebuilding canonical causal dataset and events...")
    dataset, _splits, _folds, events = build_phase5_screening_inputs(
        omni_fmt,
        omni_lst,
    )
    print(
        "      complete: "
        f"{len(dataset):,} prediction rows; "
        f"{len(events):,} canonical events"
    )

    print("[3/6] Loading protected Phase 8 event outcomes...")
    outcomes = pd.read_csv(outcomes_path)
    print(f"      complete: {len(outcomes):,} protected event rows")

    print("[4/6] Reconstructing Phase 9.3 event context...")
    context = event_context_table(events, outcomes, dataset)
    print(f"      complete: {len(context):,} event-context rows")

    print("[5/6] Building pre-onset physical-state diagnostics...")
    event_state = pre_onset_physical_state_table(
        context,
        dataset,
    )
    detected_missed = detected_vs_missed_physical_summary(
        event_state
    )
    conditional_recall = conditional_recall_summary(event_state)
    feature_shift = conditional_2025_feature_shift(event_state)
    print("      complete")

    print("[6/6] Writing diagnostic outputs...")
    event_state.to_csv(
        output_dir / "pre_onset_physical_state.csv",
        index=False,
    )
    detected_missed.to_csv(
        output_dir / "detected_vs_missed_physical_state.csv",
        index=False,
    )
    conditional_recall.to_csv(
        output_dir / "conditional_2025_recall.csv",
        index=False,
    )
    feature_shift.to_csv(
        output_dir / "conditional_2025_feature_shift.csv",
        index=False,
    )
    print("      complete")

    print()
    print("Phase 9.4 pre-onset physical-state diagnostics complete.")
    print("No model fitting performed.")
    print("No probability generation performed.")
    print("No threshold search performed.")
    print("No feature selection performed.")
    print("Official Phase 8 result unchanged.")
    print()
    print("Conditional event recall:")
    print(conditional_recall.to_string(index=False))
    print()
    print(
        "Largest absolute IQR-scaled 2025 shifts "
        "(descriptive only):"
    )

    ranked = feature_shift.dropna(
        subset=["iqr_scaled_median_shift"]
    ).copy()
    ranked["abs_shift"] = ranked[
        "iqr_scaled_median_shift"
    ].abs()
    ranked = ranked.sort_values(
        ["context_stratum", "abs_shift"],
        ascending=[True, False],
    )

    top = (
        ranked.groupby("context_stratum", sort=False)
        .head(5)
        .drop(columns="abs_shift")
    )

    if top.empty:
        print("No finite IQR-scaled shifts available.")
    else:
        print(top.to_string(index=False))

    print()
    print(f"Results written to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--omni-fmt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--omni-lst",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--phase9-1-dir",
        type=Path,
        default=Path(
            "results/phase9/operational_temporal"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/phase9/pre_onset_physical_state"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_phase9_4(
        args.omni_fmt,
        args.omni_lst,
        args.phase9_1_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
