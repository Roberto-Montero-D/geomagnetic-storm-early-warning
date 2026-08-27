"""Phase 9.3 post-hoc event-context and recurrence diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK_HOURS = (24, 48, 72)
ACTIVE_KP_THRESHOLD = 4.0
ACTIVE_LOOKBACK_HOURS = 12


def _events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "start_time", "end_time"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing canonical columns: {sorted(missing)}")
    out = events.copy()
    out["start_time"] = pd.to_datetime(out["start_time"], errors="raise")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="raise")
    return out.sort_values("start_time").reset_index(drop=True)


def _kp_lag_column(dataset: pd.DataFrame) -> str:
    candidates = ("kp_lag_1h", "kp_lag_1", "kp_lag1")
    for col in candidates:
        if col in dataset.columns:
            return col
    raise ValueError(
        "dataset must contain the frozen 1-hour Kp lag column "
        f"(tried {candidates})."
    )


def event_context_table(
    events: pd.DataFrame,
    event_outcomes: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Build one descriptive recurrence/context row per protected event."""
    events = _events(events)
    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise TypeError("dataset index must be DatetimeIndex.")
    kp_col = _kp_lag_column(dataset)

    required = {"event_id", "detected", "year"}
    missing = required - set(event_outcomes.columns)
    if missing:
        raise ValueError(f"event outcomes missing columns: {sorted(missing)}")

    outcomes = event_outcomes[["event_id", "detected", "year"]].copy()
    scoped = events.loc[events["event_id"].isin(outcomes["event_id"])].copy()
    scoped = scoped.merge(outcomes, on="event_id", how="inner", validate="one_to_one")

    all_starts = events["start_time"].to_numpy()
    rows = []
    for event in scoped.itertuples(index=False):
        prior = events.loc[events["start_time"] < event.start_time]
        previous = prior.iloc[-1] if not prior.empty else None

        row = {
            "event_id": event.event_id,
            "start_time": event.start_time,
            "year": int(event.year),
            "detected": bool(event.detected),
            "hours_since_previous_event_start": (
                np.nan if previous is None
                else float(
                    (event.start_time - previous["start_time"])
                    / pd.Timedelta(hours=1)
                )
            ),
            "hours_since_previous_event_end": (
                np.nan if previous is None
                else float(
                    (event.start_time - previous["end_time"])
                    / pd.Timedelta(hours=1)
                )
            ),
        }

        for hours in LOOKBACK_HOURS:
            lo = event.start_time - pd.Timedelta(hours=hours)
            row[f"prior_events_{hours}h"] = int(
                ((events["start_time"] >= lo)
                 & (events["start_time"] < event.start_time)).sum()
            )

        lo = event.start_time - pd.Timedelta(hours=ACTIVE_LOOKBACK_HOURS)
        background = pd.to_numeric(
            dataset.loc[
                (dataset.index >= lo) & (dataset.index < event.start_time),
                kp_col,
            ],
            errors="coerce",
        ).dropna()

        row["background_kp_median_12h"] = (
            np.nan if background.empty else float(background.median())
        )
        row["background_kp_max_12h"] = (
            np.nan if background.empty else float(background.max())
        )
        row["background_kp_active_hours_12h"] = int(
            (background >= ACTIVE_KP_THRESHOLD).sum()
        )
        row["background_kp_active_fraction_12h"] = (
            np.nan if background.empty
            else float((background >= ACTIVE_KP_THRESHOLD).mean())
        )
        rows.append(row)

    return pd.DataFrame(rows)


def recurrence_group_summary(context: pd.DataFrame) -> pd.DataFrame:
    """Compare detected/missed events without inferential testing."""
    metrics = [
        "hours_since_previous_event_start",
        "hours_since_previous_event_end",
        "prior_events_24h",
        "prior_events_48h",
        "prior_events_72h",
        "background_kp_median_12h",
        "background_kp_max_12h",
        "background_kp_active_hours_12h",
        "background_kp_active_fraction_12h",
    ]
    rows = []
    for detected, label in ((True, "detected"), (False, "missed")):
        group = context.loc[context["detected"] == detected]
        for metric in metrics:
            x = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append({
                "group": label,
                "metric": metric,
                "n": int(len(x)),
                "median": np.nan if x.empty else float(x.median()),
                "p25": np.nan if x.empty else float(x.quantile(.25)),
                "p75": np.nan if x.empty else float(x.quantile(.75)),
            })
    return pd.DataFrame(rows)


def yearly_recurrence_summary(context: pd.DataFrame) -> pd.DataFrame:
    """Year-level descriptive recurrence burden and recall."""
    rows = []
    for year, group in context.groupby("year", sort=True):
        n = len(group)
        clustered_24 = group["prior_events_24h"] > 0
        clustered_72 = group["prior_events_72h"] > 0
        active = group["background_kp_active_hours_12h"] > 0
        rows.append({
            "year": int(year),
            "n_events": int(n),
            "event_recall": float(group["detected"].mean()) if n else np.nan,
            "events_with_prior_event_24h": int(clustered_24.sum()),
            "fraction_with_prior_event_24h": float(clustered_24.mean()) if n else np.nan,
            "events_with_prior_event_72h": int(clustered_72.sum()),
            "fraction_with_prior_event_72h": float(clustered_72.mean()) if n else np.nan,
            "events_with_active_kp_12h": int(active.sum()),
            "fraction_with_active_kp_12h": float(active.mean()) if n else np.nan,
            "median_background_kp_12h": float(group["background_kp_median_12h"].median()),
        })
    return pd.DataFrame(rows)


def clustered_vs_isolated_recall(context: pd.DataFrame) -> pd.DataFrame:
    """Recall by predeclared recurrence/context strata."""
    definitions = {
        "isolated_72h": context["prior_events_72h"] == 0,
        "recurrent_72h": context["prior_events_72h"] > 0,
        "no_active_kp_12h": context["background_kp_active_hours_12h"] == 0,
        "active_kp_12h": context["background_kp_active_hours_12h"] > 0,
    }
    rows = []
    for name, mask in definitions.items():
        group = context.loc[mask]
        n = len(group)
        detected = int(group["detected"].sum())
        rows.append({
            "stratum": name,
            "n_events": int(n),
            "n_detected": detected,
            "n_missed": int(n - detected),
            "event_recall": np.nan if n == 0 else float(detected / n),
        })
    return pd.DataFrame(rows)
