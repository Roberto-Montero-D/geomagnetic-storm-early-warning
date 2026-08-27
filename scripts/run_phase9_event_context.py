"""Run Phase 9.3 event-context / recurrence diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from scripts.run_phase5_screening import build_phase5_screening_inputs
from src.final_test.diagnostics_recurrence import (
    event_context_table,
    recurrence_group_summary,
    yearly_recurrence_summary,
    clustered_vs_isolated_recall,
)


def run_phase9_3(omni_fmt, omni_lst, phase9_1_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, _splits, _folds, events = build_phase5_screening_inputs(
        omni_fmt, omni_lst
    )
    outcomes = pd.read_csv(Path(phase9_1_dir) / "event_outcomes.csv")

    context = event_context_table(events, outcomes, dataset)
    groups = recurrence_group_summary(context)
    yearly = yearly_recurrence_summary(context)
    strata = clustered_vs_isolated_recall(context)

    context.to_csv(output_dir / "event_context.csv", index=False)
    groups.to_csv(output_dir / "detected_vs_missed_recurrence.csv", index=False)
    yearly.to_csv(output_dir / "yearly_recurrence_summary.csv", index=False)
    strata.to_csv(output_dir / "recurrence_strata_recall.csv", index=False)

    print("Phase 9.3 event-context / recurrence diagnostics complete.")
    print("No model fitting performed.")
    print("No threshold search performed.")
    print("No feature selection performed.")
    print(f"Event rows: {len(context)}")
    print(f"Detected: {int(context['detected'].sum())}")
    print(f"Missed: {int((~context['detected']).sum())}")
    print()
    print(strata.to_string(index=False))
    print()
    print(yearly.to_string(index=False))
    print(f"Results written to: {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--omni-fmt", type=Path, required=True)
    p.add_argument("--omni-lst", type=Path, required=True)
    p.add_argument(
        "--phase9-1-dir",
        type=Path,
        default=Path("results/phase9/operational_temporal"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase9/event_context_recurrence"),
    )
    a = p.parse_args()
    run_phase9_3(a.omni_fmt, a.omni_lst, a.phase9_1_dir, a.output_dir)


if __name__ == "__main__":
    main()
