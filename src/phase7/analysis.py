"""Phase 7 cross-experiment development analysis.

Reporting-only utilities for the six frozen Phase 7 experiments.
The protected Final Test is never scored here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.phase7.contract import (
    PHASE7_EXPERIMENTS,
    PHASE7_EXPERIMENT_IDS,
    PHASE7_MAX_FAR_PER_DAY,
    PHASE7_MODEL_CONFIG_ID,
    PHASE7_PRIMARY_CONTROL_ID,
    get_phase7_experiment,
)

PHASE7_ANALYSIS_COLUMNS = (
    "experiment_id",
    "storm_threshold",
    "horizon_hours",
    "is_primary_control",
    "selected_config_id",
    "selected_threshold",
    "event_recall",
    "n_events",
    "n_detected_events",
    "n_alert_episodes",
    "false_alarm_episodes",
    "valid_exposure_hours",
    "exposure_days",
    "far_per_day",
    "target_prevalence",
    "walk_forward_1_threshold",
    "walk_forward_2_threshold",
    "stability_threshold_min",
    "stability_threshold_max",
    "stability_threshold_count",
    "protected_final_test_scored",
)

_REQUIRED_SUMMARY_KEYS = (
    "experiment_id",
    "storm_threshold",
    "horizon_hours",
    "selected_config_id",
    "selected_threshold",
    "event_recall",
    "n_events",
    "n_detected_events",
    "n_alert_episodes",
    "false_alarm_episodes",
    "valid_exposure_hours",
    "far_per_day",
    "fold_selected_thresholds",
    "stability_thresholds",
    "protected_final_test_scored",
)


def _as_float(value, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _as_nonnegative_int(value, *, name: str) -> int:
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _validate_registered_identity(summary: Mapping[str, object]) -> None:
    experiment_id = str(summary["experiment_id"])
    spec = get_phase7_experiment(experiment_id)
    threshold = _as_float(summary["storm_threshold"], name="storm_threshold")
    horizon = int(summary["horizon_hours"])

    if threshold != float(spec.threshold):
        raise ValueError(
            f"{experiment_id} storm threshold differs from frozen registry."
        )
    if horizon != int(spec.horizon_hours):
        raise ValueError(
            f"{experiment_id} horizon differs from frozen registry."
        )


def _validate_summary(summary: Mapping[str, object]) -> None:
    missing = [key for key in _REQUIRED_SUMMARY_KEYS if key not in summary]
    if missing:
        raise ValueError(
            "Phase 7 summary is missing required key(s): " + ", ".join(missing)
        )

    _validate_registered_identity(summary)
    experiment_id = str(summary["experiment_id"])

    if str(summary["selected_config_id"]) != PHASE7_MODEL_CONFIG_ID:
        raise ValueError(f"{experiment_id} model configuration drifted.")
    if bool(summary["protected_final_test_scored"]):
        raise ValueError(f"{experiment_id} reports protected Final Test scoring.")

    n_events = _as_nonnegative_int(summary["n_events"], name="n_events")
    n_detected = _as_nonnegative_int(
        summary["n_detected_events"], name="n_detected_events"
    )
    n_alerts = _as_nonnegative_int(
        summary["n_alert_episodes"], name="n_alert_episodes"
    )
    n_false = _as_nonnegative_int(
        summary["false_alarm_episodes"], name="false_alarm_episodes"
    )
    exposure_hours = _as_nonnegative_int(
        summary["valid_exposure_hours"], name="valid_exposure_hours"
    )

    if n_detected > n_events:
        raise ValueError(f"{experiment_id} detected events exceed eligible events.")
    if n_false > n_alerts:
        raise ValueError(f"{experiment_id} false alarms exceed alert episodes.")
    if exposure_hours <= 0:
        raise ValueError(f"{experiment_id} valid exposure must be positive.")
    if n_events == 0:
        raise ValueError(f"{experiment_id} has no eligible development events.")

    event_recall = _as_float(summary["event_recall"], name="event_recall")
    expected_recall = n_detected / n_events
    if not np.isclose(event_recall, expected_recall, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{experiment_id} event recall does not equal "
            "detected_events / eligible_events."
        )

    far = _as_float(summary["far_per_day"], name="far_per_day")
    expected_far = n_false / (exposure_hours / 24.0)
    if not np.isclose(far, expected_far, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{experiment_id} FAR/day does not equal "
            "false_alarm_episodes / exposure_days."
        )
    if far > PHASE7_MAX_FAR_PER_DAY + 1e-12:
        raise ValueError(f"{experiment_id} selected threshold violates FAR/day limit.")

    selected_threshold = _as_float(
        summary["selected_threshold"], name="selected_threshold"
    )
    if not 0.0 <= selected_threshold <= 1.0:
        raise ValueError(f"{experiment_id} selected threshold must lie in [0, 1].")

    fold_thresholds = summary["fold_selected_thresholds"]
    if not isinstance(fold_thresholds, Mapping):
        raise TypeError(
            f"{experiment_id} fold_selected_thresholds must be a mapping."
        )
    if set(fold_thresholds) != {"walk_forward_1", "walk_forward_2"}:
        raise ValueError(f"{experiment_id} must preserve the two frozen OOF folds.")

    for fold, threshold in fold_thresholds.items():
        if threshold is None:
            continue
        value = _as_float(threshold, name=f"{fold} threshold")
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{experiment_id} {fold} threshold must lie in [0, 1]."
            )

    stability = summary["stability_thresholds"]
    if not isinstance(stability, Sequence) or isinstance(stability, (str, bytes)):
        raise TypeError(f"{experiment_id} stability_thresholds must be a sequence.")
    for value in stability:
        threshold = _as_float(value, name="stability threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"{experiment_id} stability threshold must lie in [0, 1]."
            )


def _target_prevalence(
    experiment_id: str,
    target_prevalence: Mapping[str, float] | None,
) -> float:
    if target_prevalence is None:
        return np.nan
    if experiment_id not in target_prevalence:
        raise ValueError(f"Missing target prevalence for {experiment_id}.")
    value = _as_float(
        target_prevalence[experiment_id],
        name=f"{experiment_id} target prevalence",
    )
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{experiment_id} target prevalence must lie in [0, 1].")
    return value


def build_experiment_summary(
    summaries: Sequence[Mapping[str, object]],
    *,
    target_prevalence: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build the validated six-experiment Phase 7 development summary."""

    if not isinstance(summaries, Sequence) or isinstance(summaries, (str, bytes)):
        raise TypeError("summaries must be a sequence of mappings.")

    by_id: dict[str, Mapping[str, object]] = {}
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise TypeError("Every Phase 7 summary must be a mapping.")
        _validate_summary(summary)
        experiment_id = str(summary["experiment_id"])
        if experiment_id in by_id:
            raise ValueError(f"Duplicate Phase 7 summary: {experiment_id}.")
        by_id[experiment_id] = summary

    if set(by_id) != set(PHASE7_EXPERIMENT_IDS):
        missing = sorted(set(PHASE7_EXPERIMENT_IDS) - set(by_id))
        extra = sorted(set(by_id) - set(PHASE7_EXPERIMENT_IDS))
        raise ValueError(
            "Phase 7 analysis must contain exactly the six frozen experiments. "
            f"Missing={missing}, extra={extra}."
        )

    rows: list[dict[str, object]] = []
    for spec in PHASE7_EXPERIMENTS:
        summary = by_id[spec.experiment_id]
        fold_thresholds = summary["fold_selected_thresholds"]
        stability = tuple(float(value) for value in summary["stability_thresholds"])
        exposure_hours = int(summary["valid_exposure_hours"])

        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "storm_threshold": float(spec.threshold),
                "horizon_hours": int(spec.horizon_hours),
                "is_primary_control": bool(spec.is_primary_control),
                "selected_config_id": str(summary["selected_config_id"]),
                "selected_threshold": float(summary["selected_threshold"]),
                "event_recall": float(summary["event_recall"]),
                "n_events": int(summary["n_events"]),
                "n_detected_events": int(summary["n_detected_events"]),
                "n_alert_episodes": int(summary["n_alert_episodes"]),
                "false_alarm_episodes": int(summary["false_alarm_episodes"]),
                "valid_exposure_hours": exposure_hours,
                "exposure_days": exposure_hours / 24.0,
                "far_per_day": float(summary["far_per_day"]),
                "target_prevalence": _target_prevalence(
                    spec.experiment_id, target_prevalence
                ),
                "walk_forward_1_threshold": (
                    None
                    if fold_thresholds["walk_forward_1"] is None
                    else float(fold_thresholds["walk_forward_1"])
                ),
                "walk_forward_2_threshold": (
                    None
                    if fold_thresholds["walk_forward_2"] is None
                    else float(fold_thresholds["walk_forward_2"])
                ),
                "stability_threshold_min": np.nan if not stability else min(stability),
                "stability_threshold_max": np.nan if not stability else max(stability),
                "stability_threshold_count": len(stability),
                "protected_final_test_scored": False,
            }
        )

    table = pd.DataFrame(rows, columns=PHASE7_ANALYSIS_COLUMNS)
    validate_phase7_analysis(table)
    return table


def validate_phase7_analysis(table: pd.DataFrame) -> None:
    """Validate the final reporting table against the frozen contract."""

    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame.")
    if tuple(table.columns) != PHASE7_ANALYSIS_COLUMNS:
        raise ValueError(
            "Phase 7 analysis columns differ from the frozen reporting contract."
        )
    if table["experiment_id"].tolist() != list(PHASE7_EXPERIMENT_IDS):
        raise ValueError("Phase 7 analysis experiment order differs from registry.")
    if table["experiment_id"].duplicated().any():
        raise ValueError("Phase 7 analysis contains duplicate experiments.")
    if table["protected_final_test_scored"].astype(bool).any():
        raise ValueError("Protected Final Test entered Phase 7 analysis.")

    controls = table.loc[
        table["is_primary_control"].astype(bool), "experiment_id"
    ].tolist()
    if controls != [PHASE7_PRIMARY_CONTROL_ID]:
        raise ValueError("Phase 7 primary-control identity drifted.")

    for row in table.itertuples(index=False):
        spec = get_phase7_experiment(row.experiment_id)
        if float(row.storm_threshold) != float(spec.threshold):
            raise ValueError(f"{row.experiment_id} T differs from registry.")
        if int(row.horizon_hours) != int(spec.horizon_hours):
            raise ValueError(f"{row.experiment_id} H differs from registry.")
        if int(row.n_detected_events) > int(row.n_events):
            raise ValueError(f"{row.experiment_id} detected events exceed events.")

        expected_recall = int(row.n_detected_events) / int(row.n_events)
        if not np.isclose(
            float(row.event_recall), expected_recall, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{row.experiment_id} global recall is inconsistent.")

        expected_far = int(row.false_alarm_episodes) / float(row.exposure_days)
        if not np.isclose(
            float(row.far_per_day), expected_far, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{row.experiment_id} FAR/day is inconsistent.")
        if float(row.far_per_day) > PHASE7_MAX_FAR_PER_DAY + 1e-12:
            raise ValueError(f"{row.experiment_id} violates the FAR/day budget.")


def build_horizon_comparison(experiment_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the controlled T=5 horizon comparison."""

    validate_phase7_analysis(experiment_summary)
    result = experiment_summary.loc[
        experiment_summary["storm_threshold"].eq(5.0)
    ].copy()
    result = result.sort_values("horizon_hours", kind="mergesort").reset_index(
        drop=True
    )
    expected = ["t5_h3", "t5_h6", "t5_h12", "t5_h24"]
    if result["experiment_id"].tolist() != expected:
        raise AssertionError("Frozen T=5 horizon comparison drifted.")
    return result


def build_severity_comparison(experiment_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the controlled H=6 severity comparison."""

    validate_phase7_analysis(experiment_summary)
    result = experiment_summary.loc[
        experiment_summary["horizon_hours"].eq(6)
    ].copy()
    result = result.sort_values("storm_threshold", kind="mergesort").reset_index(
        drop=True
    )
    expected = ["t5_h6", "t6_h6", "t7_h6"]
    if result["experiment_id"].tolist() != expected:
        raise AssertionError("Frozen H=6 severity comparison drifted.")
    return result


def build_fold_diagnostics(experiment_summary: pd.DataFrame) -> pd.DataFrame:
    """Return long-form fold-specific selected-threshold diagnostics."""

    validate_phase7_analysis(experiment_summary)
    rows: list[dict[str, object]] = []
    for row in experiment_summary.itertuples(index=False):
        rows.extend(
            [
                {
                    "experiment_id": row.experiment_id,
                    "fold": "walk_forward_1",
                    "selected_threshold": row.walk_forward_1_threshold,
                },
                {
                    "experiment_id": row.experiment_id,
                    "fold": "walk_forward_2",
                    "selected_threshold": row.walk_forward_2_threshold,
                },
            ]
        )
    return pd.DataFrame(
        rows,
        columns=("experiment_id", "fold", "selected_threshold"),
    )
