"""Phase 9.1 operational temporal decomposition.

Post-hoc diagnostics only. This module never fits a model or searches/selects
a threshold. It decomposes the immutable Phase 8 alert episodes and canonical
event truth by calendar year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "start_time", "end_time", "boundary_status"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing canonical columns: {sorted(missing)}")
    out = events.copy()
    out["start_time"] = pd.to_datetime(out["start_time"], errors="raise")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="raise")
    if out["event_id"].duplicated().any():
        raise ValueError("canonical event_id values must be unique.")
    return out


def _validate_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    required = {
        "first_alert_time",
        "classification",
        "associated_event_id",
        "lead_time",
    }
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"episodes missing canonical columns: {sorted(missing)}")
    out = episodes.copy()
    out["first_alert_time"] = pd.to_datetime(
        out["first_alert_time"], errors="raise"
    )
    out["lead_time"] = pd.to_timedelta(out["lead_time"], errors="coerce")
    allowed = {"false_alarm", "early_detection", "late_detection"}
    if not out["classification"].isin(allowed).all():
        raise ValueError("unknown alert classification.")
    return out


def yearly_operational_decomposition(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    start: pd.Timestamp = pd.Timestamp("2022-01-01"),
    end_exclusive: pd.Timestamp = pd.Timestamp("2026-01-01"),
) -> pd.DataFrame:
    """Compute exact calendar-year event recall and false alarms/day.

    Events are assigned to the calendar year of canonical ``start_time``.
    False alarms are assigned to the year of ``first_alert_time``.
    FAR/day uses the exact duration of each calendar-year slice.
    """

    events = _validate_events(events)
    episodes = _validate_episodes(episodes)

    events = events.loc[
        (events["start_time"] >= start)
        & (events["start_time"] < end_exclusive)
    ].copy()
    episodes = episodes.loc[
        (episodes["first_alert_time"] >= start)
        & (episodes["first_alert_time"] < end_exclusive)
    ].copy()

    detection_eps = episodes.loc[
        episodes["classification"].isin(["early_detection", "late_detection"])
        & episodes["associated_event_id"].notna()
    ].copy()

    rows = []
    for year in range(start.year, end_exclusive.year):
        y0 = max(start, pd.Timestamp(f"{year}-01-01"))
        y1 = min(end_exclusive, pd.Timestamp(f"{year + 1}-01-01"))
        exposure_days = float((y1 - y0) / pd.Timedelta(days=1))

        year_events = events.loc[
            (events["start_time"] >= y0) & (events["start_time"] < y1)
        ]
        event_ids = set(year_events["event_id"].tolist())

        associated = detection_eps.loc[
            detection_eps["associated_event_id"].isin(event_ids)
        ]
        detected_ids = set(associated["associated_event_id"].tolist())
        early_ids = set(
            associated.loc[
                associated["classification"] == "early_detection",
                "associated_event_id",
            ].tolist()
        )
        late_ids = set(
            associated.loc[
                associated["classification"] == "late_detection",
                "associated_event_id",
            ].tolist()
        )

        year_eps = episodes.loc[
            (episodes["first_alert_time"] >= y0)
            & (episodes["first_alert_time"] < y1)
        ]
        false_alarms = int(
            (year_eps["classification"] == "false_alarm").sum()
        )
        n_events = len(year_events)
        n_detected = len(detected_ids)

        early_leads = associated.loc[
            associated["classification"] == "early_detection",
            "lead_time",
        ].dropna()

        rows.append(
            {
                "year": year,
                "exposure_days": exposure_days,
                "n_events": n_events,
                "n_detected_events": n_detected,
                "n_missed_events": n_events - n_detected,
                "event_recall": (
                    np.nan if n_events == 0 else n_detected / n_events
                ),
                "n_events_with_early_detection": len(early_ids),
                "n_events_with_late_detection": len(late_ids),
                "n_events_with_both_early_and_late": len(early_ids & late_ids),
                "alert_episodes": int(len(year_eps)),
                "false_alarm_episodes": false_alarms,
                "far_per_day": false_alarms / exposure_days,
                "early_detection_episodes": int(
                    (year_eps["classification"] == "early_detection").sum()
                ),
                "late_detection_episodes": int(
                    (year_eps["classification"] == "late_detection").sum()
                ),
                "median_early_lead_hours": (
                    np.nan
                    if early_leads.empty
                    else float(
                        early_leads.median() / pd.Timedelta(hours=1)
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def event_outcomes(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    start: pd.Timestamp = pd.Timestamp("2022-01-01"),
    end_exclusive: pd.Timestamp = pd.Timestamp("2026-01-01"),
) -> pd.DataFrame:
    """Return one immutable diagnostic row per protected canonical event."""

    events = _validate_events(events)
    episodes = _validate_episodes(episodes)
    scoped = events.loc[
        (events["start_time"] >= start)
        & (events["start_time"] < end_exclusive)
    ].copy()

    detections = episodes.loc[
        episodes["classification"].isin(["early_detection", "late_detection"])
        & episodes["associated_event_id"].notna()
    ].copy()

    rows = []
    for event in scoped.itertuples(index=False):
        assoc = detections.loc[
            detections["associated_event_id"] == event.event_id
        ]
        early = assoc.loc[assoc["classification"] == "early_detection"]
        late = assoc.loc[assoc["classification"] == "late_detection"]
        early_leads = early["lead_time"].dropna()

        rows.append(
            {
                "event_id": event.event_id,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "year": event.start_time.year,
                "detected": not assoc.empty,
                "early_detected": not early.empty,
                "late_detected": not late.empty,
                "n_detection_episodes": int(len(assoc)),
                "first_detection_time": (
                    pd.NaT if assoc.empty else assoc["first_alert_time"].min()
                ),
                "max_early_lead_hours": (
                    np.nan
                    if early_leads.empty
                    else float(
                        early_leads.max() / pd.Timedelta(hours=1)
                    )
                ),
            }
        )

    return pd.DataFrame(rows)
