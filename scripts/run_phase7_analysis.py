"""Build the frozen Phase 7 development comparison artifacts.

This runner is reporting-only. It reads already-generated Phase 6/7 OOF and
threshold-selection artifacts, injects the frozen t5_h6 Phase 6 control, and
writes controlled horizon/severity comparison tables.

No model is fit, no threshold is re-selected, and the protected Final Test is
never read or scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase7.analysis import (
    build_experiment_summary,
    build_fold_diagnostics,
    build_horizon_comparison,
    build_severity_comparison,
)
from src.phase7.contract import (
    PHASE7_EXPERIMENTS,
    PHASE7_EXPERIMENT_IDS,
    PHASE7_MODEL_CONFIG_ID,
    PHASE7_PRIMARY_CONTROL_ID,
)


NON_CONTROL_IDS = tuple(
    experiment.experiment_id
    for experiment in PHASE7_EXPERIMENTS
    if not experiment.is_primary_control
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")

    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def _selected_threshold_row(
    table: pd.DataFrame,
    selected_threshold: float,
) -> pd.Series:
    required = {
        "threshold",
        "event_recall",
        "n_events",
        "n_detected_events",
        "n_alert_episodes",
        "n_false_alarm_episodes",
        "valid_exposure_hours",
        "false_alarm_rate_per_day",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            "Threshold table missing required columns: "
            + ", ".join(missing)
        )

    threshold = pd.to_numeric(
        table["threshold"],
        errors="raise",
    ).to_numpy(dtype=float)

    matches = np.isclose(
        threshold,
        float(selected_threshold),
        rtol=0.0,
        atol=1e-12,
    )

    if int(matches.sum()) != 1:
        raise ValueError(
            "Selected threshold must map to exactly one threshold-table row."
        )

    return table.loc[matches].iloc[0]


def build_phase6_control_summary(
    phase6_dir: Path,
) -> dict:
    """Convert frozen Phase 6 artifacts into the Phase 7 summary schema."""

    selection = _read_json(
        phase6_dir / "phase6_selection_summary.json"
    )

    if selection.get("protected_final_test_scored") is not False:
        raise ValueError(
            "Phase 6 control does not explicitly preserve Final Test isolation."
        )

    if selection.get("selected_config_id") != PHASE7_MODEL_CONFIG_ID:
        raise ValueError(
            "Phase 6 control model differs from frozen Phase 7 model."
        )

    selected_threshold = selection.get("selected_threshold")
    if selected_threshold is None:
        raise ValueError("Phase 6 control has no selected threshold.")

    curve = _read_csv(
        phase6_dir / "global_threshold_curve.csv"
    )
    selected = _selected_threshold_row(
        curve,
        float(selected_threshold),
    )

    spec = next(
        experiment
        for experiment in PHASE7_EXPERIMENTS
        if experiment.experiment_id == PHASE7_PRIMARY_CONTROL_ID
    )

    fold_thresholds = selection.get("fold_selected_thresholds")
    if not isinstance(fold_thresholds, dict):
        raise ValueError(
            "Phase 6 fold_selected_thresholds must be a JSON object."
        )

    stability = selection.get("stability_thresholds")
    if not isinstance(stability, list):
        raise ValueError(
            "Phase 6 stability_thresholds must be a JSON list."
        )

    return {
        "experiment_id": spec.experiment_id,
        "storm_threshold": float(spec.threshold),
        "horizon_hours": int(spec.horizon_hours),
        "selected_config_id": selection["selected_config_id"],
        "selected_threshold": float(selected_threshold),
        "event_recall": float(selected["event_recall"]),
        "n_events": int(selected["n_events"]),
        "n_detected_events": int(selected["n_detected_events"]),
        "n_alert_episodes": int(selected["n_alert_episodes"]),
        "false_alarm_episodes": int(
            selected["n_false_alarm_episodes"]
        ),
        "valid_exposure_hours": int(
            selected["valid_exposure_hours"]
        ),
        "far_per_day": float(
            selected["false_alarm_rate_per_day"]
        ),
        "fold_selected_thresholds": {
            str(fold): (
                None if threshold is None else float(threshold)
            )
            for fold, threshold in fold_thresholds.items()
        },
        "stability_thresholds": [
            float(value)
            for value in stability
        ],
        "protected_final_test_scored": False,
    }


def load_phase7_non_control_summaries(
    experiments_dir: Path,
) -> list[dict]:
    """Load the five frozen non-control selection summaries."""

    execution = _read_json(
        experiments_dir / "phase7_execution_summary.json"
    )

    if execution.get("protected_final_test_scored") is not False:
        raise ValueError(
            "Phase 7 execution summary does not preserve Final Test isolation."
        )
    if execution.get("primary_control_executed") is not False:
        raise ValueError(
            "Phase 7 non-control run unexpectedly executed the primary control."
        )
    if execution.get("selected_config_id") != PHASE7_MODEL_CONFIG_ID:
        raise ValueError(
            "Phase 7 execution model differs from frozen contract."
        )

    executed = tuple(execution.get("executed_experiments", ()))
    if executed != NON_CONTROL_IDS:
        raise ValueError(
            "Phase 7 executed experiment set differs from frozen registry."
        )

    summaries = execution.get("experiments")
    if not isinstance(summaries, list):
        raise ValueError(
            "Phase 7 execution summary must contain an experiments list."
        )

    by_id = {
        str(summary.get("experiment_id")): summary
        for summary in summaries
        if isinstance(summary, dict)
    }

    if set(by_id) != set(NON_CONTROL_IDS):
        raise ValueError(
            "Phase 7 execution summary must contain exactly the five "
            "non-control experiments."
        )

    # Require the per-experiment selection artifact to agree with the
    # cross-experiment execution summary, so the analysis cannot silently use
    # a partially overwritten directory.
    result: list[dict] = []

    for experiment_id in NON_CONTROL_IDS:
        aggregate = by_id[experiment_id]
        individual = _read_json(
            experiments_dir
            / experiment_id
            / "selection_summary.json"
        )

        if aggregate != individual:
            raise ValueError(
                f"{experiment_id} aggregate and individual summaries differ."
            )

        result.append(individual)

    return result


def _oof_target_prevalence(path: Path) -> float:
    table = _read_csv(path)

    if "target" not in table.columns:
        raise ValueError(f"OOF artifact has no target column: {path}")

    target = pd.to_numeric(
        table["target"],
        errors="raise",
    )

    if target.isna().any():
        raise ValueError(f"OOF target contains missing values: {path}")

    unique = set(target.astype(int).unique().tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"OOF target is not binary: {path}")

    return float(target.mean())


def load_target_prevalence(
    phase6_dir: Path,
    experiments_dir: Path,
) -> dict[str, float]:
    """Read OOF target prevalence for all six frozen experiments."""

    prevalence = {
        PHASE7_PRIMARY_CONTROL_ID: _oof_target_prevalence(
            phase6_dir / "oof_predictions.csv"
        )
    }

    for experiment_id in NON_CONTROL_IDS:
        prevalence[experiment_id] = _oof_target_prevalence(
            experiments_dir
            / experiment_id
            / "oof_predictions.csv"
        )

    if set(prevalence) != set(PHASE7_EXPERIMENT_IDS):
        raise AssertionError(
            "Target prevalence map does not contain all six experiments."
        )

    return prevalence


def run_phase7_analysis(
    phase6_dir: Path,
    experiments_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Build and write the six-experiment development-only analysis."""

    control = build_phase6_control_summary(phase6_dir)
    non_control = load_phase7_non_control_summaries(
        experiments_dir
    )
    prevalence = load_target_prevalence(
        phase6_dir,
        experiments_dir,
    )

    summary = build_experiment_summary(
        [*non_control, control],
        target_prevalence=prevalence,
    )
    horizon = build_horizon_comparison(summary)
    severity = build_severity_comparison(summary)
    folds = build_fold_diagnostics(summary)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir / "experiment_summary.csv",
        index=False,
    )
    horizon.to_csv(
        output_dir / "horizon_comparison.csv",
        index=False,
    )
    severity.to_csv(
        output_dir / "severity_comparison.csv",
        index=False,
    )
    folds.to_csv(
        output_dir / "fold_threshold_diagnostics.csv",
        index=False,
    )

    payload = {
        "selected_config_id": PHASE7_MODEL_CONFIG_ID,
        "experiments": list(PHASE7_EXPERIMENT_IDS),
        "primary_control_id": PHASE7_PRIMARY_CONTROL_ID,
        "protected_final_test_scored": False,
        "comparison_policy": {
            "horizon": "Compare T=5 only while varying H.",
            "severity": "Compare H=6 only while varying T.",
            "cross_task_ranking_authorized": False,
        },
        "target_prevalence": {
            experiment_id: float(prevalence[experiment_id])
            for experiment_id in PHASE7_EXPERIMENT_IDS
        },
        "horizon_experiments": (
            horizon["experiment_id"].astype(str).tolist()
        ),
        "severity_experiments": (
            severity["experiment_id"].astype(str).tolist()
        ),
    }

    with (
        output_dir / "phase7_analysis_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen six-experiment Phase 7 development comparison "
            "from already-generated Phase 6/7 artifacts."
        )
    )
    parser.add_argument(
        "--phase6-dir",
        type=Path,
        default=Path("results/phase6/threshold_selection"),
    )
    parser.add_argument(
        "--phase7-experiments-dir",
        type=Path,
        default=Path("results/phase7/experiments"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase7/analysis"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    summary = run_phase7_analysis(
        args.phase6_dir,
        args.phase7_experiments_dir,
        args.output_dir,
    )

    print("Phase 7 development analysis complete.")
    print(f"Results written to: {args.output_dir}")
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
                "n_detected_events",
                "n_events",
                "far_per_day",
                "target_prevalence",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
