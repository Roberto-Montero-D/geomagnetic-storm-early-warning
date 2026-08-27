"""Run Phase 9 post-hoc diagnostics from immutable Phase 8 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.final_test.diagnostics import (
    calibration_table,
    lead_time_summary,
    yearly_episode_diagnostics,
    yearly_probability_diagnostics,
)


def run_phase9_diagnostics(
    phase8_dir: Path,
    output_dir: Path,
) -> None:
    phase8_dir = Path(phase8_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        phase8_dir / "final_test_predictions.csv",
        parse_dates=["prediction_time"],
    ).set_index("prediction_time")

    episodes = pd.read_csv(
        phase8_dir / "final_test_alert_episodes.csv",
    )

    yearly_probability_diagnostics(
        predictions
    ).to_csv(
        output_dir / "yearly_probability_diagnostics.csv",
        index=False,
    )

    yearly_episode_diagnostics(
        episodes
    ).to_csv(
        output_dir / "yearly_episode_diagnostics.csv",
        index=False,
    )

    calibration_table(
        predictions
    ).to_csv(
        output_dir / "calibration_table.csv",
        index=False,
    )

    lead_time_summary(
        episodes
    ).to_csv(
        output_dir / "lead_time_summary.csv",
        index=False,
    )

    print("Phase 9 post-hoc diagnostics complete.")
    print("No threshold search performed.")
    print("No model fitting performed.")
    print("Official Phase 8 metrics unchanged.")
    print(f"Results written to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run descriptive post-hoc diagnostics over the immutable "
            "Phase 8 Final Test artifacts."
        )
    )
    parser.add_argument(
        "--phase8-dir",
        type=Path,
        default=Path("results/phase8/final_test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase9/diagnostics"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_phase9_diagnostics(
        args.phase8_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
