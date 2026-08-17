# Geomagnetic Storm Event Definition

**Protocol:** `MASTER_PROTOCOL_v1.1`
**Status:** Specification complete — implementation pending

---

## 1. Purpose

This document defines how the hourly Kp time series is converted into discrete geomagnetic storm events.

The event definition is deterministic and independent of:

* model predictions;
* probability thresholds;
* alert episodes;
* model selection;
* and model performance.

`MASTER_PROTOCOL_v1.1.md` is the authoritative source for this definition.

---

## 2. Primary Parameters

The primary experiment uses:

```text
T = 5
Z = 6 hours
```

where:

* `T` is the geomagnetic storm threshold;
* `Z` is the number of consecutive below-threshold hours required to terminate an event.

These parameters are frozen by the master protocol.

---

## 3. Event Start

A geomagnetic storm event starts at the first valid hourly observation satisfying:

```text
Kp >= T
```

For the primary experiment:

```text
Kp >= 5
```

The timestamp of that observation is the event start time:

```text
storm_start = first hour where Kp >= T
```

---

## 4. Event Continuation

Once an event has started, it remains active until the termination condition has been satisfied.

Temporary periods where:

```text
Kp < T
```

do not immediately terminate the event.

If Kp returns to or exceeds the threshold before `Z` consecutive valid below-threshold hours have occurred, the same event continues.

Therefore, elevated periods separated by fewer than `Z` consecutive below-threshold hours belong to the same storm event.

---

## 5. Event Termination

An event terminates only after:

```text
Z = 6
```

consecutive valid hourly observations satisfying:

```text
Kp < T
```

The storm end is defined as the last active storm hour before the terminating sequence of `Z` consecutive below-threshold observations.

Conceptually:

```text
Kp >= T
Kp >= T
Kp < T   ← first terminating hour
Kp < T
Kp < T
Kp < T
Kp < T
Kp < T   ← termination condition confirmed
```

The six below-threshold hours establish that the previous storm event has ended.

They are not part of the storm event itself.

---

## 6. Event Separation

A new threshold crossing belongs to a new independent event only after the previous event has satisfied the complete termination rule.

Therefore:

```text
fewer than 6 consecutive hours below T
→ same event
```

while:

```text
6 consecutive valid hours below T
→ previous event terminated
→ next Kp >= T starts a new event
```

This rule prevents short fluctuations below the threshold from fragmenting a single storm into multiple events.

---

## 7. Missing Observations

A missing timestamp or missing Kp observation must **not** be interpreted as:

```text
Kp < T
```

and therefore must not automatically contribute to the `Z = 6` termination sequence.

For example:

```text
12:00   Kp = 4
13:00   missing
14:00   Kp = 4
```

does not represent three consecutive valid below-threshold observations.

The event implementation must explicitly verify hourly continuity when evaluating the termination condition.

The final missing-data behavior must remain consistent with the project Data Contract.

---

## 8. Dataset Boundaries

### Event active at the beginning of the dataset

If the dataset begins while:

```text
Kp >= T
```

the implementation cannot prove that the observed threshold crossing is the true physical beginning of the storm.

This boundary condition must be identified explicitly rather than silently treated as a confirmed event start.

### Event active at the end of the dataset

If an event begins but the dataset ends before the full termination condition can be observed, the event must be treated as right-censored at the dataset boundary.

The implementation must not invent future below-threshold observations to close the event.

Boundary handling must be deterministic and tested.

---

## 9. Canonical Event Representation

Each identified storm event should be represented by a structured record containing at least:

```text
event_id
start_time
end_time
threshold
```

The implementation may include additional metadata when useful, such as:

```text
peak_kp
duration
left_censored
right_censored
```

These additional fields do not change the event definition.

The same event identifiers should be reusable across:

* target construction;
* alert association;
* evaluation;
* error analysis;
* and case studies.

---

## 10. Relationship to the Forecast Target

The event definition and target definition are related but conceptually separate.

The event definition answers:

> When did geomagnetic storm events occur?

The target answers:

> Will a geomagnetic storm occur within the next `H` hours?

For the primary experiment:

```text
H = 6 hours
T = 5
```

Future Kp information used for target construction must remain separate from feature construction according to the Data Contract.

---

## 11. Required Unit Tests

Before event identification is used by the modeling pipeline, the implementation must include tests for at least:

```text
test_basic_event_start
test_event_termination_after_z_hours
test_same_event_when_gap_less_than_z
test_new_event_after_z_hour_separation
test_consecutive_storm_hours
test_missing_timestamp_does_not_count_as_below_threshold
test_left_dataset_boundary
test_right_dataset_boundary
```

### Minimum behavioral cases

**Case 1 — Basic event**

```text
4 4 5 6 5 4 4 4 4 4 4
    ↑
 event start
```

The threshold crossing starts one event.

**Case 2 — Insufficient separation**

If elevated activity returns before six consecutive valid below-threshold hours have occurred, both active periods belong to the same event.

**Case 3 — Complete separation**

After six consecutive valid below-threshold hours, a subsequent threshold crossing starts a new event.

**Case 4 — Missing observation**

A missing hourly observation cannot be counted as one of the six required below-threshold hours.

---

## 12. Canonical Implementation

The event definition must be implemented once and reused throughout the project.

The canonical implementation belongs under:

```text
src/definitions/
```

Evaluation code, notebooks, models, and analysis scripts must not create independent interpretations of the event definition.

---

## 13. Frozen Rule

For the primary system:

```text
T = 5
Z = 6 hours
```

These values must not be modified based on model performance or final-test results.

Any alternative experiment must be explicitly identified as such and must not silently redefine the primary storm-event definition.
