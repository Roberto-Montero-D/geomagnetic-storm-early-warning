# Operational Alert Definition

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Specification complete — implementation pending

---

## 1. Purpose

This document defines how hourly model probabilities are converted into operational alert episodes and how those episodes are associated with geomagnetic storm events. The definition is deterministic and independent of model family.

`MASTER_PROTOCOL_v1.3.md` is the authoritative source.

## 2. Primary Parameters

```text
H = 6 hours
C = 3 hours
maximum FAR/day = 0.2
```

The probability threshold `tau` is selected using the protocol-defined OOF procedure.

## 3. Hourly Alert

At prediction time `t`, an alert is generated when:

```text
P(storm within H hours) >= tau
```

Prediction time `t` is the operational forecast issuance time. All features used to produce the probability at `t` must independently satisfy the Data Contract.

## 4. Alert Episode Construction

For `C = 3h`:

```text
gap <= 3h -> same episode
gap > 3h  -> new episode
```

Missing prediction timestamps must not silently create or merge episodes.

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

Optional metadata may include number of alert hours, maximum probability, threshold, and duration.

## 6. Alert-to-Storm Association

`storm_start` and `storm_end` come exclusively from the canonical event representation in `docs/event_definition.md`. Alert classification must not independently reconstruct storm boundaries from raw hourly OMNI Kp rows.

Each alert episode is associated with at most one event.

## 7. Early Detection

```text
storm_start - H <= first_alert_time < storm_start
```

The left boundary is inclusive: an alert exactly at `storm_start - H` is a valid Early Detection.

```text
lead_time = storm_start - first_alert_time
```

## 8. Late Detection

```text
storm_start <= first_alert_time <= storm_end
```

An alert exactly at `storm_start` is a Late Detection, not an Early Detection. Late Detections count toward Event Recall but remain separate from Early Detections.

## 9. False Alarm

An alert episode is a False Alarm when it cannot be validly associated with any storm as either an Early Detection or a Late Detection under the frozen association rules.

A too-early alert (`first_alert_time < storm_start - H`) is not a detection of that candidate storm, but chronological association must still determine whether it belongs to another event.

False alarms are counted per episode, not per hourly prediction.

## 10. Multiple Alerts Associated With One Storm

If multiple episodes qualify as Early Detections for the same storm, the earliest qualifying `first_alert_time` is used for storm-level detection and lead-time calculation.

If no Early Detection exists but one or more Late Detections exist, the earliest qualifying Late Detection is used.

Additional qualifying episodes do not increase Event Recall.

## 11. Event Recall

```text
Event Recall = (Early-detected events + Late-detected events)
               / total evaluable storm events
```

Hourly target recall must not substitute for Event Recall.

## 12. False Alarm Rate per Day

```text
FAR/day = number of false alert episodes / evaluation duration in days
```

The exact denominator treatment when prediction timestamps are invalid or unavailable must be frozen during Phase 0 before threshold optimization.

Primary constraint:

```text
FAR/day <= 0.2
```

## 13. Lead Time

```text
lead_time = storm_start - first_alert_time
0 < lead_time <= H
```

Lead-time statistics use storm-level Early Detections. Late Detections are not included in the primary Early Detection lead-time distribution.

## 14. Threshold Selection

Select the minimum OOF threshold satisfying `FAR/day <= 0.2`. If no candidate threshold satisfies the constraint, the model is recorded as failing the operational constraint; the constraint must not be relaxed retrospectively.

## 15. Threshold Stability Analysis

The diagnostic range `0.15 <= FAR/day <= 0.20` assesses stability only. It never replaces the global OOF-selected operational threshold.

## 16. Missing Predictions and Temporal Gaps

A missing prediction timestamp is not automatically equivalent to `alert = false`. Missing-time behavior must be defined by the evaluation pipeline and remain consistent with temporal-integrity rules.

## 17. Dataset Boundaries

Boundary behavior must be deterministic, must not invent predictions outside available data, and must not cause duplicate alert counting.

## 18. Required Unit Tests

```text
test_single_alert_episode
test_consecutive_alerts_same_episode
test_alert_gap_equal_to_c
test_alert_gap_greater_than_c
test_early_detection
test_early_detection_left_boundary_inclusive
test_late_detection
test_storm_start_is_late_not_early
test_false_alarm
test_multiple_alerts_single_storm
test_multiple_early_episodes_uses_earliest
test_early_episode_preferred_over_late_episode
test_multiple_storms
test_event_recall_not_double_counted
test_false_alarm_counted_per_episode
test_lead_time
test_missing_prediction_timestamp
test_alert_dataset_boundaries
test_episode_cannot_detect_multiple_storms_ambiguously
test_no_threshold_satisfies_far_constraint
```

## 19. Canonical Implementation

Alert construction/classification must be implemented once under `src/definitions/`. Threshold selection and operational metrics must consume these canonical episodes rather than implement independent grouping logic.

## 20. Frozen Primary Rule

```text
H = 6 hours
C = 3 hours
maximum FAR/day = 0.2
```

The threshold `tau` is selected exclusively from development/OOF information and must not be modified after observing final-test performance.
