"""Execute the five frozen non-control Phase 7 experiments.

This runner performs development-only OOF prediction and experiment-specific
operational threshold recalibration. The protected Final Test is never scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from scripts.run_phase5_screening import (
    _progress,
    build_phase5_screening_inputs,
)
from src.data.kp import build_kp_intervals
from src.data.omni import load_omni
from src.evaluation.phase6_thresholds import (
    DEFAULT_STABILITY_MIN_FAR_PER_DAY,
)
from src.evaluation.threshold_selection import (
    DEFAULT_THRESHOLD_GRID,
)
from src.phase7.contract import (
    PHASE7_EXPERIMENTS,
    PHASE7_MAX_FAR_PER_DAY,
    PHASE7_MODEL_CONFIG_ID,
    PHASE7_PRIMARY_CONTROL_ID,
)
from src.phase7.oof import (
    assert_phase7_oof_is_development_only,
    generate_phase7_oof_predictions,
)
from src.phase7.thresholds import optimize_phase7_threshold


PHASE7_EXECUTION_EXPERIMENTS = tuple(
    experiment
    for experiment in PHASE7_EXPERIMENTS
    if experiment.experiment_id != PHASE7_PRIMARY_CONTROL_ID
)


def _json_number(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _selected_row(threshold_result):
    threshold = threshold_result.selected_threshold

    if threshold is None:
        return None

    table = threshold_result.global_threshold_table
    rows = table.loc[
        table["threshold"].astype(float).eq(float(threshold))
    ]

    if len(rows) != 1:
        raise AssertionError(
            "Selected Phase 7 threshold does not map to exactly "
            "one global threshold-table row."
        )

    return rows.iloc[0]


def write_phase7_experiment_artifacts(
    experiment,
    oof_result,
    threshold_result,
    output_dir: Path,
) -> dict:
    """Write one standardized development-only Phase 7 result set."""

    experiment_dir = output_dir / experiment.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    oof_result.table.to_csv(
        experiment_dir / "oof_predictions.csv",
        index=True,
    )

    threshold_result.global_threshold_table.to_csv(
        experiment_dir / "global_threshold_curve.csv",
        index=False,
    )

    threshold_result.fold_threshold_table.to_csv(
        experiment_dir / "fold_threshold_curves.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "fold": fold,
                "selected_threshold": threshold,
            }
            for fold, threshold in (
                threshold_result.fold_selected_thresholds.items()
            )
        ],
        columns=["fold", "selected_threshold"],
    ).to_csv(
        experiment_dir / "fold_selected_thresholds.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "threshold": list(
                threshold_result.stability_thresholds
            )
        }
    ).to_csv(
        experiment_dir / "stability_thresholds.csv",
        index=False,
    )

    selected = _selected_row(threshold_result)

    summary = {
        "experiment_id": experiment.experiment_id,
        "storm_threshold": float(experiment.threshold),
        "horizon_hours": int(experiment.horizon_hours),
        "selected_config_id": oof_result.config_id,
        "selected_threshold": _json_number(
            threshold_result.selected_threshold
        ),
        "max_far_per_day": float(PHASE7_MAX_FAR_PER_DAY),
        "stability_min_far_per_day": float(
            DEFAULT_STABILITY_MIN_FAR_PER_DAY
        ),
        "threshold_grid_min": float(DEFAULT_THRESHOLD_GRID[0]),
        "threshold_grid_max": float(DEFAULT_THRESHOLD_GRID[-1]),
        "threshold_grid_size": int(len(DEFAULT_THRESHOLD_GRID)),
        "oof_rows": int(len(oof_result.table)),
        "oof_folds": [
            str(value)
            for value in (
                oof_result.table["fold"]
                .drop_duplicates()
                .tolist()
            )
        ],
        "fold_selected_thresholds": {
            str(fold): _json_number(threshold)
            for fold, threshold in (
                threshold_result.fold_selected_thresholds.items()
            )
        },
        "stability_thresholds": [
            float(value)
            for value in threshold_result.stability_thresholds
        ],
        "event_recall": (
            None
            if selected is None
            else _json_number(selected["event_recall"])
        ),
        "n_events": (
            None
            if selected is None
            else int(selected["n_events"])
        ),
        "n_detected_events": (
            None
            if selected is None
            else int(selected["n_detected_events"])
        ),
        "n_alert_episodes": (
            None
            if selected is None
            else int(selected["n_alert_episodes"])
        ),        
        "false_alarm_episodes": (
            None
            if selected is None
            else int(selected["n_false_alarm_episodes"])
        ),
        "valid_exposure_hours": (
            None
            if selected is None
            else int(selected["valid_exposure_hours"])
        ),
        "far_per_day": (
            None
            if selected is None
            else _json_number(
                selected["false_alarm_rate_per_day"]
            )
        ),
        "protected_final_test_scored": False,
    }

    with (
        experiment_dir / "selection_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return summary


def run_phase7_experiments(
    fmt_path: Path,
    lst_path: Path,
    output_dir: Path,
):
    """Execute all five pre-authorized non-control experiments."""

    if tuple(
        experiment.experiment_id
        for experiment in PHASE7_EXECUTION_EXPERIMENTS
    ) != (
        "t5_h3",
        "t5_h12",
        "t5_h24",
        "t6_h6",
        "t7_h6",
    ):
        raise AssertionError(
            "Phase 7 execution experiment set drifted."
        )

    _progress(
        "[Phase 7 1/4] Building canonical development inputs..."
    )
    start = perf_counter()

    (
        dataset,
        splits,
        folds,
        _,
    ) = build_phase5_screening_inputs(
        fmt_path,
        lst_path,
    )

    _progress(
        "      canonical development inputs complete "
        f"[{perf_counter() - start:.1f} s]"
    )

    _progress(
        "[Phase 7 2/4] Building canonical Kp intervals..."
    )
    start = perf_counter()

    omni = load_omni(
        fmt_path,
        lst_path,
    )
    kp_intervals = build_kp_intervals(omni)

    _progress(
        f"      complete: {len(kp_intervals):,} intervals "
        f"[{perf_counter() - start:.1f} s]"
    )

    _progress(
        "[Phase 7 3/4] Executing five frozen "
        "non-control experiments..."
    )

    summaries = []

    for number, experiment in enumerate(
        PHASE7_EXECUTION_EXPERIMENTS,
        start=1,
    ):
        experiment_start = perf_counter()

        _progress(
            f"    [{number}/5] {experiment.experiment_id}: "
            f"T={experiment.threshold:g}, "
            f"H={experiment.horizon_hours} h"
        )

        _progress("          generating OOF predictions...")

        oof_result = generate_phase7_oof_predictions(
            dataset,
            kp_intervals,
            folds,
            splits,
            experiment,
            progress=True,
        )

        assert_phase7_oof_is_development_only(
            oof_result,
            splits,
        )

        if oof_result.config_id != PHASE7_MODEL_CONFIG_ID:
            raise AssertionError(
                "Phase 7 model configuration drifted."
            )

        _progress(
            "          recalibrating operational threshold..."
        )

        threshold_result = optimize_phase7_threshold(
            oof_result,
            kp_intervals,
            experiment,
            thresholds=DEFAULT_THRESHOLD_GRID,
            max_far_per_day=PHASE7_MAX_FAR_PER_DAY,
            stability_min_far_per_day=(
                DEFAULT_STABILITY_MIN_FAR_PER_DAY
            ),
            progress=True,
        )

        summary = write_phase7_experiment_artifacts(
            experiment,
            oof_result,
            threshold_result,
            output_dir,
        )
        summaries.append(summary)

        _progress(
            "          complete: "
            f"tau={summary['selected_threshold']}, "
            f"recall={summary['event_recall']}, "
            f"FAR/day={summary['far_per_day']} "
            f"[{perf_counter() - experiment_start:.1f} s]"
        )

    _progress(
        "[Phase 7 4/4] Writing cross-experiment summary..."
    )

    summary_table = pd.DataFrame(summaries)

    summary_table.to_csv(
        output_dir / "phase7_experiment_summary.csv",
        index=False,
    )

    with (
        output_dir / "phase7_execution_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_config_id": PHASE7_MODEL_CONFIG_ID,
                "executed_experiments": [
                    experiment.experiment_id
                    for experiment in PHASE7_EXECUTION_EXPERIMENTS
                ],
                "primary_control_executed": False,
                "protected_final_test_scored": False,
                "experiments": summaries,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return summary_table


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the five frozen non-control Phase 7 "
            "development-only experiments."
        )
    )

    parser.add_argument(
        "--omni-fmt",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--omni-lst",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase7/experiments"),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    summary = run_phase7_experiments(
        args.omni_fmt,
        args.omni_lst,
        args.output_dir,
    )

    print()
    print("Phase 7 non-control experiment execution complete.")
    print(f"Results written to: {args.output_dir}")
    print(f"Frozen configuration: {PHASE7_MODEL_CONFIG_ID}")
    print("Protected Final Test scored: False")
    print()
    print(
        summary[
            [
                "experiment_id",
                "storm_threshold",
                "horizon_hours",
                "selected_threshold",
                "event_recall",
                "far_per_day",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()