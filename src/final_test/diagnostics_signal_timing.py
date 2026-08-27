"""Phase 9.5 onset-centered signal-timing diagnostics.

Post-hoc / exploratory only.

This module aligns already-frozen causal predictor states and immutable Phase 8
probabilities to canonical storm onset. It does not fit models, generate new
probabilities, search thresholds, select features, or alter Phase 8 results.

Offsets are defined in OPERATIONAL PREDICTION TIME:

    relative_hour = prediction_time - storm_start

The primary diagnostic range is -12 h through -1 h. Thus every row is a
prediction made strictly before canonical event onset. Predictor values at each
prediction timestamp already obey the project's frozen information cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TRAJECTORY_START_HOUR = -12
TRAJECTORY_END_HOUR = -1

# Small, physically interpretable set chosen from already-frozen predictors.
# This is not feature selection; no ranking or downstream modeling is allowed.
SIGNAL_FEATURES = (
    "bz_gsm",
    "bt",
    "speed",
    "flow_pressure",
    "kp_lag_1h",
    "bz_neg_x_speed",
)

EARLY_WINDOW = (-12, -7)
WARNING_WINDOW = (-6, -1)


def _validate_context(context: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id",
        "start_time",
        "year",
        "detected",
        "prior_events_72h",
        "background_kp_active_hours_12h",
    }
    missing = required - set(context.columns)
    if missing:
        raise ValueError(
            f"event context missing columns: {sorted(missing)}"
        )

    out = context.copy()
    out["start_time"] = pd.to_datetime(
        out["start_time"], errors="raise"
    )
    out["detected"] = out["detected"].astype(bool)

    if out["event_id"].duplicated().any():
        raise ValueError("event context must contain one row per event.")

    return out.sort_values("start_time").reset_index(drop=True)


def _validate_dataset(dataset: pd.DataFrame) -> None:
    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise TypeError("dataset index must be DatetimeIndex.")
    if dataset.index.hasnans:
        raise ValueError("dataset index must not contain NaT.")
    if dataset.index.has_duplicates:
        raise ValueError("dataset index must be unique.")
    if not dataset.index.is_monotonic_increasing:
        raise ValueError(
            "dataset index must be monotonically increasing."
        )

    missing = set(SIGNAL_FEATURES) - set(dataset.columns)
    if missing:
        raise ValueError(
            "canonical dataset missing frozen signal columns: "
            f"{sorted(missing)}"
        )


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()

    if "prediction_time" in work.columns:
        work["prediction_time"] = pd.to_datetime(
            work["prediction_time"], errors="raise"
        )
        work = work.set_index("prediction_time")

    if not isinstance(work.index, pd.DatetimeIndex):
        raise TypeError(
            "Phase 8 predictions must use a DatetimeIndex or "
            "contain prediction_time."
        )
    if work.index.has_duplicates:
        raise ValueError(
            "Phase 8 prediction timestamps must be unique."
        )
    if "probability" not in work.columns:
        raise ValueError(
            "Phase 8 predictions missing probability column."
        )

    probability = pd.to_numeric(
        work["probability"], errors="raise"
    )
    if probability.isna().any():
        raise ValueError(
            "Phase 8 prediction probability contains missing values."
        )
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError(
            "Phase 8 probabilities must lie in [0, 1]."
        )

    return work.sort_index()


def _context_stratum(row: pd.Series) -> str:
    recurrent = int(row["prior_events_72h"]) > 0
    active = int(row["background_kp_active_hours_12h"]) > 0

    if recurrent and active:
        return "recurrent_active"
    if recurrent:
        return "recurrent_quiet"
    if active:
        return "isolated_active"
    return "isolated_quiet"


def onset_aligned_event_trajectory(
    context: pd.DataFrame,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    start_hour: int = TRAJECTORY_START_HOUR,
    end_hour: int = TRAJECTORY_END_HOUR,
) -> pd.DataFrame:
    """Build one row per event and pre-onset prediction-time offset."""
    if start_hour >= 0 or end_hour >= 0:
        raise ValueError(
            "Phase 9.5 trajectory offsets must remain pre-onset."
        )
    if start_hour > end_hour:
        raise ValueError(
            "start_hour must be <= end_hour."
        )

    context = _validate_context(context)
    _validate_dataset(dataset)
    predictions = _validate_predictions(predictions)

    offsets = list(range(start_hour, end_hour + 1))
    rows: list[dict[str, object]] = []

    for _, event in context.iterrows():
        start = event["start_time"]
        stratum = _context_stratum(event)
        period = (
            "2025"
            if int(event["year"]) == 2025
            else "2022_2024"
        )
        outcome = (
            "detected" if bool(event["detected"]) else "missed"
        )

        for relative_hour in offsets:
            prediction_time = (
                start + pd.Timedelta(hours=relative_hour)
            )

            row: dict[str, object] = {
                "event_id": event["event_id"],
                "storm_start": start,
                "year": int(event["year"]),
                "detected": bool(event["detected"]),
                "outcome_group": outcome,
                "context_stratum": stratum,
                "comparison_period": period,
                "relative_hour": int(relative_hour),
                "prediction_time": prediction_time,
            }

            if prediction_time in dataset.index:
                source = dataset.loc[prediction_time]
                for feature in SIGNAL_FEATURES:
                    value = pd.to_numeric(
                        pd.Series([source[feature]]),
                        errors="coerce",
                    ).iloc[0]
                    row[feature] = (
                        np.nan
                        if pd.isna(value)
                        else float(value)
                    )
            else:
                for feature in SIGNAL_FEATURES:
                    row[feature] = np.nan

            if prediction_time in predictions.index:
                row["probability"] = float(
                    predictions.at[prediction_time, "probability"]
                )
            else:
                row["probability"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def grouped_trajectory_summary(
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize onset-aligned trajectories without inferential testing."""
    required = {
        "relative_hour",
        "outcome_group",
        "context_stratum",
        "comparison_period",
        *SIGNAL_FEATURES,
        "probability",
    }
    missing = required - set(trajectory.columns)
    if missing:
        raise ValueError(
            f"trajectory missing columns: {sorted(missing)}"
        )

    value_columns = (*SIGNAL_FEATURES, "probability")
    rows: list[dict[str, object]] = []

    grouping_specs = (
        ("outcome", ["outcome_group"]),
        ("period", ["comparison_period"]),
        (
            "outcome_period",
            ["outcome_group", "comparison_period"],
        ),
        (
            "context_outcome",
            ["context_stratum", "outcome_group"],
        ),
        (
            "context_period",
            ["context_stratum", "comparison_period"],
        ),
        (
            "context_outcome_period",
            [
                "context_stratum",
                "outcome_group",
                "comparison_period",
            ],
        ),
    )

    for grouping_name, keys in grouping_specs:
        grouped = trajectory.groupby(
            keys + ["relative_hour"],
            dropna=False,
            sort=True,
        )

        for group_key, frame in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            base = {
                "grouping": grouping_name,
                "relative_hour": int(
                    frame["relative_hour"].iloc[0]
                ),
            }
            for key, value in zip(keys, group_key[:-1]):
                base[key] = value

            # group_key's final element is relative_hour because it was
            # appended to keys in groupby.
            if len(keys) == 1:
                base[keys[0]] = group_key[0]
            else:
                for key, value in zip(keys, group_key[:len(keys)]):
                    base[key] = value

            for metric in value_columns:
                x = pd.to_numeric(
                    frame[metric], errors="coerce"
                ).dropna()
                rows.append(
                    {
                        **base,
                        "metric": metric,
                        "n": int(len(x)),
                        "median": (
                            np.nan
                            if x.empty
                            else float(x.median())
                        ),
                        "p25": (
                            np.nan
                            if x.empty
                            else float(x.quantile(0.25))
                        ),
                        "p75": (
                            np.nan
                            if x.empty
                            else float(x.quantile(0.75))
                        ),
                    }
                )

    return pd.DataFrame(rows)


def timing_window_contrast(
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each event's early and warning-window physical state.

    This is a descriptive within-event contrast only. No threshold or
    significance testing is performed.
    """
    rows: list[dict[str, object]] = []

    value_columns = (*SIGNAL_FEATURES, "probability")

    for event_id, frame in trajectory.groupby(
        "event_id", sort=False
    ):
        meta = frame.iloc[0]

        early = frame.loc[
            frame["relative_hour"].between(
                EARLY_WINDOW[0],
                EARLY_WINDOW[1],
            )
        ]
        warning = frame.loc[
            frame["relative_hour"].between(
                WARNING_WINDOW[0],
                WARNING_WINDOW[1],
            )
        ]

        for metric in value_columns:
            e = pd.to_numeric(
                early[metric], errors="coerce"
            ).dropna()
            w = pd.to_numeric(
                warning[metric], errors="coerce"
            ).dropna()

            early_median = (
                np.nan if e.empty else float(e.median())
            )
            warning_median = (
                np.nan if w.empty else float(w.median())
            )
            change = (
                np.nan
                if (
                    np.isnan(early_median)
                    or np.isnan(warning_median)
                )
                else float(warning_median - early_median)
            )

            rows.append(
                {
                    "event_id": event_id,
                    "storm_start": meta["storm_start"],
                    "year": int(meta["year"]),
                    "outcome_group": meta["outcome_group"],
                    "context_stratum": meta["context_stratum"],
                    "comparison_period": meta[
                        "comparison_period"
                    ],
                    "metric": metric,
                    "early_window": "-12h_to_-7h",
                    "warning_window": "-6h_to_-1h",
                    "n_early": int(len(e)),
                    "n_warning": int(len(w)),
                    "median_early": early_median,
                    "median_warning": warning_median,
                    "warning_minus_early": change,
                }
            )

    return pd.DataFrame(rows)


def grouped_timing_contrast_summary(
    event_contrast: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize within-event timing contrasts by protected groups."""
    rows: list[dict[str, object]] = []

    grouping_specs = (
        ("outcome", ["outcome_group"]),
        (
            "outcome_period",
            ["outcome_group", "comparison_period"],
        ),
        (
            "context_outcome",
            ["context_stratum", "outcome_group"],
        ),
        (
            "context_outcome_period",
            [
                "context_stratum",
                "outcome_group",
                "comparison_period",
            ],
        ),
    )

    for grouping_name, keys in grouping_specs:
        for group_key, frame in event_contrast.groupby(
            keys, dropna=False, sort=True
        ):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            key_values = dict(zip(keys, group_key))

            for metric, metric_frame in frame.groupby(
                "metric", sort=True
            ):
                x = pd.to_numeric(
                    metric_frame["warning_minus_early"],
                    errors="coerce",
                ).dropna()

                rows.append(
                    {
                        "grouping": grouping_name,
                        **key_values,
                        "metric": metric,
                        "n_events": int(len(x)),
                        "median_warning_minus_early": (
                            np.nan
                            if x.empty
                            else float(x.median())
                        ),
                        "p25_warning_minus_early": (
                            np.nan
                            if x.empty
                            else float(x.quantile(0.25))
                        ),
                        "p75_warning_minus_early": (
                            np.nan
                            if x.empty
                            else float(x.quantile(0.75))
                        ),
                    }
                )

    return pd.DataFrame(rows)


def trajectory_coverage_summary(
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    """Report data coverage by relative hour."""
    value_columns = (*SIGNAL_FEATURES, "probability")
    rows: list[dict[str, object]] = []

    for relative_hour, frame in trajectory.groupby(
        "relative_hour", sort=True
    ):
        row: dict[str, object] = {
            "relative_hour": int(relative_hour),
            "n_event_rows": int(len(frame)),
        }
        for metric in value_columns:
            row[f"{metric}_available"] = int(
                pd.to_numeric(
                    frame[metric], errors="coerce"
                ).notna().sum()
            )
            row[f"{metric}_available_fraction"] = float(
                pd.to_numeric(
                    frame[metric], errors="coerce"
                ).notna().mean()
            )
        rows.append(row)

    return pd.DataFrame(rows)
