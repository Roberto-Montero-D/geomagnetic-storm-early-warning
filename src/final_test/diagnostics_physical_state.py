"""Phase 9.4 post-hoc pre-onset physical-state diagnostics.

This module is descriptive only. It does not fit models, generate new
probabilities, select features, or search thresholds.

For each protected Phase 8 event, frozen causal predictors are summarized over
the six-hour operational warning window immediately preceding canonical event
onset. The resulting event-level table is then used to compare detected versus
missed events and to describe conditional 2025 shifts relative to 2022--2024.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PRE_ONSET_HOURS = 6

# These are already-frozen causal predictors from the canonical dataset.
# They are diagnostic views only, not new candidate features.
PHYSICAL_FEATURES = (
    "bz_gsm",
    "bt",
    "speed",
    "density",
    "flow_pressure",
    "kp_lag_1h",
    "bz_gsm_delta_1h",
    "bz_gsm_delta_3h",
    "bz_gsm_slope_3h",
    "speed_delta_1h",
    "speed_delta_3h",
    "speed_slope_3h",
    "flow_pressure_delta_1h",
    "flow_pressure_delta_3h",
    "flow_pressure_slope_3h",
    "bz_gsm_persist_lt_m5h",
    "bz_gsm_persist_lt_m10h",
    "bz_gsm_persist_lt_m15h",
    "speed_persist_gt_500h",
    "speed_persist_gt_600h",
)


def _validate_inputs(
    context: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
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

    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise TypeError("dataset index must be DatetimeIndex.")
    if dataset.index.has_duplicates:
        raise ValueError("dataset index must be unique.")
    if not dataset.index.is_monotonic_increasing:
        raise ValueError(
            "dataset index must be monotonically increasing."
        )

    missing_features = set(PHYSICAL_FEATURES) - set(dataset.columns)
    if missing_features:
        raise ValueError(
            "canonical dataset missing frozen physical feature columns: "
            f"{sorted(missing_features)}"
        )

    out = context.copy()
    out["start_time"] = pd.to_datetime(
        out["start_time"], errors="raise"
    )
    out["detected"] = out["detected"].astype(bool)
    return out.sort_values("start_time").reset_index(drop=True)


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


def pre_onset_physical_state_table(
    context: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    pre_onset_hours: int = PRE_ONSET_HOURS,
) -> pd.DataFrame:
    """Return one causal pre-onset physical-state row per protected event.

    The diagnostic window is:
        [event_start - pre_onset_hours, event_start)

    Every dataset row in that interval is already a causal prediction-time
    feature vector under the frozen information cutoff. The aggregation is
    retrospective diagnostic analysis only and is never fed back to a model.
    """
    if pre_onset_hours <= 0:
        raise ValueError("pre_onset_hours must be positive.")

    context = _validate_inputs(context, dataset)
    rows: list[dict[str, object]] = []

    base_columns = list(context.columns)

    for _, event in context.iterrows():
        start = event["start_time"]
        lo = start - pd.Timedelta(hours=pre_onset_hours)

        window = dataset.loc[
            (dataset.index >= lo) & (dataset.index < start),
            list(PHYSICAL_FEATURES),
        ]

        row = event.to_dict()
        row["context_stratum"] = _context_stratum(event)
        row["comparison_period"] = (
            "2025" if int(event["year"]) == 2025 else "2022_2024"
        )
        row["pre_onset_window_hours"] = int(pre_onset_hours)
        row["pre_onset_prediction_rows"] = int(len(window))

        for feature in PHYSICAL_FEATURES:
            x = pd.to_numeric(
                window[feature], errors="coerce"
            )
            valid = x.dropna()

            row[f"{feature}__n"] = int(len(valid))
            row[f"{feature}__last"] = (
                np.nan if valid.empty else float(valid.iloc[-1])
            )
            row[f"{feature}__median"] = (
                np.nan if valid.empty else float(valid.median())
            )
            row[f"{feature}__min"] = (
                np.nan if valid.empty else float(valid.min())
            )
            row[f"{feature}__max"] = (
                np.nan if valid.empty else float(valid.max())
            )

        rows.append(row)

    ordered_prefix = base_columns + [
        "context_stratum",
        "comparison_period",
        "pre_onset_window_hours",
        "pre_onset_prediction_rows",
    ]

    out = pd.DataFrame(rows)
    remaining = [
        c for c in out.columns if c not in ordered_prefix
    ]
    return out.loc[:, ordered_prefix + remaining]


def detected_vs_missed_physical_summary(
    event_state: pd.DataFrame,
) -> pd.DataFrame:
    """Describe physical-state metrics for detected and missed events."""
    metric_columns = [
        c
        for c in event_state.columns
        if c.endswith("__last") or c.endswith("__median")
    ]

    rows: list[dict[str, object]] = []

    for detected, label in (
        (True, "detected"),
        (False, "missed"),
    ):
        group = event_state.loc[
            event_state["detected"] == detected
        ]

        for metric in metric_columns:
            x = pd.to_numeric(
                group[metric], errors="coerce"
            ).dropna()

            rows.append(
                {
                    "group": label,
                    "metric": metric,
                    "n": int(len(x)),
                    "median": (
                        np.nan if x.empty else float(x.median())
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


def conditional_recall_summary(
    event_state: pd.DataFrame,
) -> pd.DataFrame:
    """Compare 2025 recall with 2022--2024 inside matched context strata."""
    rows: list[dict[str, object]] = []

    strata = (
        "all_events",
        "isolated_quiet",
        "isolated_active",
        "recurrent_quiet",
        "recurrent_active",
    )

    for stratum in strata:
        scoped = (
            event_state
            if stratum == "all_events"
            else event_state.loc[
                event_state["context_stratum"] == stratum
            ]
        )

        for period in ("2022_2024", "2025"):
            group = scoped.loc[
                scoped["comparison_period"] == period
            ]

            n = int(len(group))
            detected = (
                int(group["detected"].sum()) if n else 0
            )

            rows.append(
                {
                    "context_stratum": stratum,
                    "comparison_period": period,
                    "n_events": n,
                    "n_detected": detected,
                    "n_missed": int(n - detected),
                    "event_recall": (
                        np.nan if n == 0 else float(detected / n)
                    ),
                }
            )

    return pd.DataFrame(rows)


def conditional_2025_feature_shift(
    event_state: pd.DataFrame,
) -> pd.DataFrame:
    """Describe 2025 physical shifts inside matched context strata.

    No significance tests are performed.

    iqr_scaled_median_shift =
        (median_2025 - median_2022_2024) / historical_IQR

    when the historical IQR is positive.
    """
    metric_columns = [
        c
        for c in event_state.columns
        if c.endswith("__last") or c.endswith("__median")
    ]

    rows: list[dict[str, object]] = []

    strata = (
        "all_events",
        "isolated_quiet",
        "isolated_active",
        "recurrent_quiet",
        "recurrent_active",
    )

    for stratum in strata:
        scoped = (
            event_state
            if stratum == "all_events"
            else event_state.loc[
                event_state["context_stratum"] == stratum
            ]
        )

        historical = scoped.loc[
            scoped["comparison_period"] == "2022_2024"
        ]
        current = scoped.loc[
            scoped["comparison_period"] == "2025"
        ]

        for metric in metric_columns:
            h = pd.to_numeric(
                historical[metric], errors="coerce"
            ).dropna()
            y = pd.to_numeric(
                current[metric], errors="coerce"
            ).dropna()

            h_median = (
                np.nan if h.empty else float(h.median())
            )
            y_median = (
                np.nan if y.empty else float(y.median())
            )
            h_iqr = (
                np.nan
                if h.empty
                else float(
                    h.quantile(0.75) - h.quantile(0.25)
                )
            )

            median_difference = (
                np.nan
                if np.isnan(h_median) or np.isnan(y_median)
                else float(y_median - h_median)
            )

            scaled = (
                np.nan
                if (
                    np.isnan(median_difference)
                    or np.isnan(h_iqr)
                    or h_iqr <= 0
                )
                else float(median_difference / h_iqr)
            )

            rows.append(
                {
                    "context_stratum": stratum,
                    "metric": metric,
                    "n_2022_2024": int(len(h)),
                    "n_2025": int(len(y)),
                    "median_2022_2024": h_median,
                    "median_2025": y_median,
                    "median_difference_2025_minus_prior": (
                        median_difference
                    ),
                    "iqr_2022_2024": h_iqr,
                    "iqr_scaled_median_shift": scaled,
                }
            )

    return pd.DataFrame(rows)
