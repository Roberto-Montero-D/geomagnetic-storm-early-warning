"""Phase 9 diagnostics over immutable Phase 8 result artifacts.

This module performs descriptive post-hoc analysis only. It does not train
models, generate probabilities, search thresholds, or replace Phase 8 metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


FIXED_CALIBRATION_EDGES = np.linspace(0.0, 1.0, 11)


def _validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"probability", "target"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"prediction artifact is missing columns: {sorted(missing)}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(
            "prediction artifact index must be a DatetimeIndex."
        )
    if frame.index.has_duplicates:
        raise ValueError(
            "prediction artifact timestamps must be unique."
        )
    if not frame.index.is_monotonic_increasing:
        raise ValueError(
            "prediction artifact timestamps must be ordered."
        )

    result = frame.copy()
    result["probability"] = pd.to_numeric(
        result["probability"],
        errors="raise",
    ).astype(float)
    result["target"] = pd.to_numeric(
        result["target"],
        errors="coerce",
    )

    if not result["probability"].between(0.0, 1.0).all():
        raise ValueError(
            "probabilities must lie in [0, 1]."
        )

    known = result["target"].notna()
    if not result.loc[known, "target"].isin([0, 1]).all():
        raise ValueError(
            "known targets must be binary."
        )

    return result


def yearly_probability_diagnostics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute descriptive row-level metrics separately for each calendar year."""

    frame = _validate_predictions(predictions)
    rows: list[dict[str, object]] = []

    for year, group in frame.groupby(frame.index.year):
        known = group["target"].notna()
        scored = group.loc[known]
        y = scored["target"].astype(int)
        p = scored["probability"].astype(float)

        if scored.empty:
            pr_auc = np.nan
            roc_auc = np.nan
            brier = np.nan
            prevalence = np.nan
        else:
            prevalence = float(y.mean())
            brier = float(brier_score_loss(y, p))
            if y.nunique() > 1:
                pr_auc = float(
                    average_precision_score(y, p)
                )
                roc_auc = float(
                    roc_auc_score(y, p)
                )
            else:
                pr_auc = np.nan
                roc_auc = np.nan

        rows.append(
            {
                "year": int(year),
                "prediction_rows": int(len(group)),
                "known_target_rows": int(known.sum()),
                "target_prevalence": prevalence,
                "mean_probability": float(
                    group["probability"].mean()
                ),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier_score": brier,
            }
        )

    return pd.DataFrame(rows)


def calibration_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build a fixed 10-bin reliability table with no learned bin boundaries."""

    frame = _validate_predictions(predictions)
    frame = frame.loc[
        frame["target"].notna()
    ].copy()

    labels = list(range(10))
    frame["calibration_bin"] = pd.cut(
        frame["probability"],
        bins=FIXED_CALIBRATION_EDGES,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows: list[dict[str, object]] = []
    for bin_id in labels:
        group = frame.loc[
            frame["calibration_bin"] == bin_id
        ]
        lower = float(
            FIXED_CALIBRATION_EDGES[bin_id]
        )
        upper = float(
            FIXED_CALIBRATION_EDGES[bin_id + 1]
        )

        rows.append(
            {
                "bin_id": bin_id,
                "lower_bound": lower,
                "upper_bound": upper,
                "n": int(len(group)),
                "mean_probability": (
                    np.nan
                    if group.empty
                    else float(
                        group["probability"].mean()
                    )
                ),
                "observed_positive_rate": (
                    np.nan
                    if group.empty
                    else float(
                        group["target"].mean()
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def yearly_episode_diagnostics(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize immutable alert episodes by first-alert calendar year."""

    required = {
        "first_alert_time",
        "classification",
        "lead_time",
    }
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(
            f"episode artifact is missing columns: {sorted(missing)}"
        )

    frame = episodes.copy()
    frame["first_alert_time"] = pd.to_datetime(
        frame["first_alert_time"],
        errors="raise",
    )
    frame["lead_time"] = pd.to_timedelta(
        frame["lead_time"],
        errors="coerce",
    )

    allowed = {
        "false_alarm",
        "early_detection",
        "late_detection",
    }
    if not frame["classification"].isin(allowed).all():
        raise ValueError(
            "episode artifact contains an unknown classification."
        )

    rows: list[dict[str, object]] = []
    for year, group in frame.groupby(
        frame["first_alert_time"].dt.year
    ):
        early_leads = group.loc[
            group["classification"] == "early_detection",
            "lead_time",
        ].dropna()

        rows.append(
            {
                "year": int(year),
                "alert_episodes": int(len(group)),
                "false_alarm_episodes": int(
                    (
                        group["classification"]
                        == "false_alarm"
                    ).sum()
                ),
                "early_detection_episodes": int(
                    (
                        group["classification"]
                        == "early_detection"
                    ).sum()
                ),
                "late_detection_episodes": int(
                    (
                        group["classification"]
                        == "late_detection"
                    ).sum()
                ),
                "median_early_lead_hours": (
                    np.nan
                    if early_leads.empty
                    else float(
                        early_leads.median()
                        / pd.Timedelta(hours=1)
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def lead_time_summary(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize early-detection lead times without changing alert semantics."""

    required = {"classification", "lead_time"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(
            f"episode artifact is missing columns: {sorted(missing)}"
        )

    leads = pd.to_timedelta(
        episodes.loc[
            episodes["classification"] == "early_detection",
            "lead_time",
        ],
        errors="coerce",
    ).dropna()

    if leads.empty:
        values = {
            "n": 0,
            "min_hours": np.nan,
            "p25_hours": np.nan,
            "median_hours": np.nan,
            "p75_hours": np.nan,
            "max_hours": np.nan,
        }
    else:
        hours = (
            leads / pd.Timedelta(hours=1)
        ).astype(float)
        values = {
            "n": int(len(hours)),
            "min_hours": float(hours.min()),
            "p25_hours": float(hours.quantile(0.25)),
            "median_hours": float(hours.median()),
            "p75_hours": float(hours.quantile(0.75)),
            "max_hours": float(hours.max()),
        }

    return pd.DataFrame([values])
