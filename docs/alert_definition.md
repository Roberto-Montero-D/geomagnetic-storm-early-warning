# Operational Alert Definition

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Implemented, tested, and frozen

## 1. Purpose

This document defines how hourly model probabilities are converted into operational alert episodes and associated with canonical geomagnetic storm events.

## 2. Primary Parameters

```text
H = 6 hours
C = 3 hours
maximum FAR/day = 0.2
```

The probability threshold `tau` is selected using the protocol-defined OOF procedure.

## 3. Hourly Alert

At prediction time `t`:

```text
P(storm within H hours) >= tau
```

generates an alert. The threshold is inclusive.

## 4. Alert Episode Construction

For `C = 3h`:

```text
gap <= 3h -> same episode
gap > 3h  -> new episode
```

Grouping uses elapsed wall-clock time between actual alert timestamps, not row adjacency.

```text
alert 10:00 -> alert 13:00 = same episode
alert 10:00 -> alert 14:00 = different episodes
```

Missing probabilities do not independently terminate an episode.

## 5. Canonical Alert Episode Representation

Each episode contains at least:

```text
alert_episode_id
first_alert_time
last_alert_time
classification
associated_event_id
lead_time
```

The implementation also records `n_alert_hours`, `max_probability`, and `threshold`.

## 6. Alert-to-Storm Association

Storm boundaries come exclusively from the canonical event representation. Association uses `first_alert_time`, considers events chronologically, and associates each episode with at most one event. Early Detection is evaluated before Late Detection.

## 7. Early Detection

```text
storm_start - H <= first_alert_time < storm_start
lead_time = storm_start - first_alert_time
```

The left boundary is inclusive.

## 8. Late Detection

For events with finite `storm_end`:

```text
storm_start <= first_alert_time <= storm_end
```

An alert exactly at `storm_start` is Late Detection.

## 9. False Alarm

An episode is a False Alarm when it cannot be associated with any event as Early or Late Detection. False alarms are counted per episode.

## 10. Multiple Alerts Associated With One Storm

Additional qualifying episodes do not increase Event Recall. For Early Detection lead time, the earliest qualifying episode is used.

## 11. Event Recall

```text
Event Recall =
    unique events with Early or Late Detection
    /
    total evaluable storm events
```

If there are no evaluable events, Event Recall is `NaN`.

## 12. False Alarm Rate per Day

```text
valid_prediction_hours =
    number of timestamps with a finite model probability

evaluation_days =
    valid_prediction_hours / 24

FAR/day =
    false_alert_episodes / evaluation_days
```

Missing probabilities reduce exposure and are not treated as negative forecasts. With zero valid prediction hours, FAR/day is `NaN`.

Primary constraint:

```text
FAR/day <= 0.2
```

## 13. Lead Time

```text
lead_time = storm_start - first_alert_time
0 < lead_time <= H
```

Lead-time statistics use one storm-level Early Detection lead time per event. Late Detections are excluded from the primary Early Detection lead-time distribution.

## 14. Threshold Selection

Select the minimum OOF threshold satisfying:

```text
FAR/day <= 0.2
```

Threshold search remains outside the canonical alert-definition module.

## 15. Threshold Stability Analysis

```text
0.15 <= FAR/day <= 0.20
```

is diagnostic only and does not replace the global OOF-selected threshold.

## 16. Missing Predictions and Temporal Gaps

A missing probability is an unknown forecast, not `alert = false`.

It does not generate an alert, does not count as a negative prediction, reduces valid prediction exposure, and does not independently terminate an episode.

Probability timestamps must be valid, unique, monotonically increasing, and aligned to whole hours. Missing timestamps are not silently reconstructed.

## 17. Dataset Boundaries and Censored Storms

No predictions or storm boundaries are invented outside available data.

For a right-censored event with:

```text
end_time = NaT
```

the Early Detection window remains evaluable because `storm_start` is known. No unbounded Late Detection interval is invented.

## 18. Tests

The canonical tests cover episode grouping, cooldown boundaries, missing probabilities, Early/Late/False association, deterministic overlapping-window association, Event Recall, lead time, exposure-based FAR/day, censored events, timestamp integrity, malformed probabilities, and empty cases.

## 19. Canonical Implementation

Implemented in:

```text
src/definitions/alerts.py
```

Tested in:

```text
tests/test_alerts.py
```

Canonical functions include:

```text
identify_alerts()
associate_alerts_with_events()
event_recall()
valid_prediction_exposure_days()
false_alarm_rate_per_day()
early_detection_lead_times()
```

Threshold search remains outside this module.

## 20. Frozen Primary Rule

```text
H = 6 hours
C = 3 hours
maximum FAR/day = 0.2
```

The threshold `tau` is selected exclusively from development/OOF information and must not be changed after observing final-test performance.
