# Operational Alert Definition

**Protocol:** `MASTER_PROTOCOL_v1.1`
**Status:** Specification complete — implementation pending

---

## 1. Purpose

This document defines how hourly model probabilities are converted into operational alert episodes and how those episodes are associated with geomagnetic storm events.

The alert definition is deterministic and independent of model family.

`MASTER_PROTOCOL_v1.1.md` is the authoritative source for this definition.

---

## 2. Primary Parameters

The primary experiment uses:

```text
H = 6 hours
C = 3 hours
maximum FAR/day = 0.2
```

where:

* `H` is the forecast horizon;
* `C` is the alert cooldown / episode-grouping interval;
* `FAR/day` is the operational false-alarm constraint.

The probability threshold `tau` is not fixed in advance.

It is selected using the out-of-fold threshold-selection procedure defined by the master protocol.

---

## 3. Hourly Alert

At each prediction time `t`, the model produces:

```text
P(storm within H hours)
```

An hourly alert is generated when:

```text
P(storm within H hours) >= tau
```

where `tau` is the currently evaluated or frozen operational threshold.

An hourly alert is not itself an independent operational warning episode.

Hourly alerts must first be grouped into alert episodes.

---

## 4. Alert Episode Construction

Alert episodes are constructed using:

```text
C = 3 hours
```

Two alert-producing timestamps belong to the same alert episode when the time between them is no greater than `C`.

Conceptually:

```text
alert ── alert ───── alert
  └──── same episode ────┘
```

provided that consecutive alert occurrences remain within the cooldown rule.

A new alert episode begins when the gap between alert occurrences exceeds `C`.

This grouping prevents repeated hourly predictions from being counted as multiple operational warnings for the same continuous alert period.

---

## 5. Canonical Alert Episode Representation

Each alert episode should contain at least:

```text
alert_episode_id
first_alert_time
last_alert_time
classification
associated_event_id
lead_time
```

Additional metadata may include:

```text
number_of_alert_hours
maximum_probability
threshold
duration
```

These fields do not change the underlying alert definition.

---

## 6. Alert-to-Storm Association

Alert episodes are evaluated relative to the storm events defined by `docs/event_definition.md`.

The primary classification is determined from the first alert time of the episode.

For a storm with:

```text
storm_start
storm_end
```

the relevant forecast window begins at:

```text
storm_start - H
```

The alert episode is classified according to its temporal relationship with the storm.

---

## 7. Early Detection

An alert episode is an **Early Detection** when:

```text
storm_start - H <= first_alert_time < storm_start
```

For the primary system:

```text
H = 6 hours
```

so a qualifying early warning occurs within the six-hour interval preceding storm onset.

The lead time is:

```text
lead_time = storm_start - first_alert_time
```

A larger positive lead time represents an earlier warning within the valid forecast horizon.

---

## 8. Late Detection

An alert episode is a **Late Detection** when:

```text
storm_start <= first_alert_time <= storm_end
```

A late detection indicates that the system identified storm conditions only after the event had already begun.

Late detections must be reported separately from successful early warnings.

Late Detections count as detected events for Event Recall, but they must remain separate from Early Detections so that Late Detection Rate and Lead Time are reported correctly.

---

## 9. False Alarm

An alert episode is a **False Alarm** when its first alert occurs too early to correspond to a future storm under the configured forecast horizon and it cannot be validly associated with another storm event.

For a candidate storm:

```text
first_alert_time < storm_start - H
```

does not constitute a valid early detection of that storm.

Alert association must consider the chronological storm sequence so that an episode is not labeled a false alarm if it validly belongs to another event.

False alarms are counted at the **alert-episode level**, not at the hourly-prediction level.

---

## 10. Multiple Alerts Associated With One Storm

A storm event may have more than one nearby alert episode.

However, event recall is binary at the storm level:

```text
storm detected
```

or:

```text
storm not detected
```

Multiple alert episodes must not cause the same storm event to be counted multiple times in event recall.

The implementation must deterministically identify the qualifying episode used for event detection and lead-time calculation according to the protocol.

---

## 11. Event Recall

Event Recall measures the fraction of storm events receiving a qualifying detection.

A storm is considered detected when it receives either:

- an Early Detection; or
- a Late Detection.

Conceptually:

```text
               detected storm events
Event Recall = ---------------------
                  total storm events

---

## 12. False Alarm Rate per Day

False Alarm Rate per day is calculated from false **alert episodes**, not individual positive hourly predictions.

Conceptually:

```text
          number of false alert episodes
FAR/day = ------------------------------
             evaluation duration in days
```

The exact evaluation duration must be calculated consistently across validation folds, OOF threshold selection, and final testing.

The primary operational constraint is:

```text
FAR/day <= 0.2
```

---

## 13. Lead Time

Lead time is calculated for a qualifying early detection as:

```text
lead_time = storm_start - first_alert_time
```

For the primary system:

```text
0 < lead_time <= H
```

with:

```text
H = 6 hours
```

Lead-time statistics should be calculated from storm-level detections rather than treating repeated alerts for the same event as independent observations.

---

## 14. Threshold Selection

The operational probability threshold `tau` is selected using temporally ordered out-of-fold predictions.

Candidate thresholds are evaluated according to the master protocol.

The primary rule is to select the minimum threshold satisfying:

```text
FAR/day <= 0.2
```

The threshold is frozen before the protected final test is evaluated.

The final test must not be used to choose or adjust `tau`.

---

## 15. Threshold Stability Analysis

The protocol also evaluates the diagnostic range:

```text
0.15 <= FAR/day <= 0.20
```

This analysis is used to assess threshold stability.

It does not replace the primary threshold-selection rule and must not be used retrospectively to choose a more favorable final-test result.

---

## 16. Missing Predictions and Temporal Gaps

Missing prediction timestamps must not silently create or merge alert episodes.

The implementation must verify temporal continuity when applying the cooldown parameter `C`.

A missing timestamp is not equivalent to:

```text
alert = false
```

unless that behavior is explicitly defined by the evaluation pipeline.

Alert grouping must remain consistent with the project's missing-data and temporal-integrity rules.

---

## 17. Dataset Boundaries

Alert episodes near the beginning or end of an evaluation period require explicit handling.

The implementation must not:

* associate an alert with an event outside the permitted evaluation logic without documenting the boundary behavior;
* invent predictions outside the available dataset;
* or truncate an episode in a way that causes duplicate alert counting.

Boundary behavior must be deterministic and covered by tests.

---

## 18. Required Unit Tests

Before operational evaluation is used for model comparison, the implementation must include tests for at least:

```text
test_single_alert_episode
test_consecutive_alerts_same_episode
test_alert_gap_equal_to_c
test_alert_gap_greater_than_c
test_early_detection
test_late_detection
test_false_alarm
test_multiple_alerts_single_storm
test_multiple_storms
test_event_recall_not_double_counted
test_false_alarm_counted_per_episode
test_lead_time
test_missing_prediction_timestamp
test_alert_dataset_boundaries
```

### Minimum behavioral cases

**Case 1 — Early detection**

```text
first alert       storm start
    |-----------------|
          <= H
```

The episode is an Early Detection.

**Case 2 — Late detection**

```text
storm start     first alert     storm end
     |--------------|--------------|
```

The episode is a Late Detection.

**Case 3 — False alarm**

An alert episode with no valid associated storm within the forecast horizon is a False Alarm.

**Case 4 — Repeated alerts**

Several alert hours or episodes associated with one storm must not increase event recall above one detection for that storm.

---

## 19. Canonical Implementation

Alert construction and classification must be implemented once and reused throughout the project.

The canonical implementation belongs under:

```text
src/definitions/
```

Threshold selection and operational metrics should consume these canonical alert episodes rather than implementing separate alert-grouping logic.

---

## 20. Frozen Primary Rule

For the primary system:

```text
H = 6 hours
C = 3 hours
maximum FAR/day = 0.2
```

The threshold `tau` is selected exclusively from development / OOF information according to the frozen protocol.

These rules must not be modified after observing final-test performance.
