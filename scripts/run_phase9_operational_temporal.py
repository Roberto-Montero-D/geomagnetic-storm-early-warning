"""Run Phase 9.1 operational temporal decomposition.

This is post-hoc descriptive analysis. It does not fit models, generate new
probabilities, or search thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.run_phase5_screening import build_phase5_screening_inputs
from src.final_test.diagnostics_operational import (
    event_outcomes,
    yearly_operational_decomposition,
)


def run_phase9_1(
    omni_fmt: Path,
    omni_lst: Path,
    phase8_dir: Path,
    output_dir: Path,
) -> None:
    phase8_dir = Path(phase8_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        _dataset,
        _splits,
        _folds,
        events,
    ) = build_phase5_screening_inputs(
        omni_fmt,
        omni_lst,
    )

    episodes = pd.read_csv(
        phase8_dir / "final_test_alert_episodes.csv"
    )

    yearly = yearly_operational_decomposition(events, episodes)
    outcomes = event_outcomes(events, episodes)

    yearly.to_csv(
        output_dir / "yearly_operational_decomposition.csv",
        index=False,
    )
    outcomes.to_csv(
        output_dir / "event_outcomes.csv",
        index=False,
    )

    print("Phase 9.1 operational temporal decomposition complete.")
    print("No threshold search performed.")
    print("No model fitting performed.")
    print("Official Phase 8 result unchanged.")
    print()
    print(yearly.to_string(index=False))
    print()
    print(f"Event rows: {len(outcomes)}")
    print(f"Detected: {int(outcomes['detected'].sum())}")
    print(f"Missed: {int((~outcomes['detected']).sum())}")
    print(
        "False-alarm episodes: "
        f"{int(yearly['false_alarm_episodes'].sum())}"
    )
    print(f"Results written to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-fmt", type=Path, required=True)
    parser.add_argument("--omni-lst", type=Path, required=True)
    parser.add_argument(
        "--phase8-dir",
        type=Path,
        default=Path("results/phase8/final_test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase9/operational_temporal"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_phase9_1(
        args.omni_fmt,
        args.omni_lst,
        args.phase8_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
