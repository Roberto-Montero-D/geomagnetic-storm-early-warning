# Geomagnetic Storm Event Definition

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Specification complete — implementation pending

---

## 1. Purpose

This document defines how retrospective canonical Kp is converted into discrete geomagnetic storm events. The definition is deterministic and independent of model predictions, probability thresholds, alert episodes, model selection, and model performance.

`MASTER_PROTOCOL_v1.3.md` is the authoritative source.

## 2. Primary Parameters

```text
T = 5
Z = 6 hours
```

`T` is the storm-condition threshold. `Z` is the number of consecutive valid below-threshold hourly ground-truth states required to terminate an event.

## 3. Event Start

Kp is interpreted from its canonical 3-hour interval representation. An event starts at the **start of the first Kp interval** satisfying:

```text
Kp >= T
```

For the primary experiment: `Kp >= 5`.

Although OMNI repeats each Kp value over three hourly rows, those rows do not represent three independent Kp observations.

## 4. Event Continuation

Once started, an event remains active until the complete termination condition is satisfied. If Kp returns to or exceeds T before Z consecutive valid below-threshold hourly ground-truth states occur, the same event continues.

## 5. Event Termination

For event segmentation, canonical Kp intervals are represented on the project's hourly evaluation timeline. A 3-hour Kp interval contributes three consecutive hourly states carrying the same retrospective Kp value.

Therefore, with complete aligned data, `Z = 6h` corresponds to two complete consecutive 3-hour Kp intervals below threshold.

The storm end is the last active storm hour before the terminating sequence. The six below-threshold states establish termination and are not part of the storm event itself.

## 6. Event Separation

```text
fewer than 6 valid consecutive hours below T -> same event
6 valid consecutive hours below T           -> previous event terminated
subsequent Kp >= T                           -> new event
```

## 7. Missing Observations

A missing timestamp, missing Kp value, or missing canonical Kp interval must not be interpreted as `Kp < T` and must not contribute to the Z-hour termination sequence. A missing canonical interval produces missing hourly ground-truth states for that interval.

## 8. Dataset Boundaries

Events at the beginning or end of the dataset must be identified explicitly. The implementation must never invent observations outside available data. Boundary behavior must be deterministic and frozen during the event-implementation stage of Phase 0 before model results are inspected.

## 9. Canonical Event Representation

Each event should contain at least:

```text
event_id
start_time
end_time
threshold
```

Optional audit metadata may include `peak_kp`, `duration`, and `boundary_status`.

## 10. Relationship to the Forecast Target

Event identification uses retrospective canonical Kp. Predictor-side `kp_asof()` is not used for event or target ground truth.

The event definition answers when storms occurred; the target asks whether storm conditions occur within the next H hours.

## 11. Required Unit Tests

```text
test_basic_event_start
test_event_start_matches_canonical_kp_interval_start
test_repeated_hourly_kp_rows_do_not_create_multiple_events
test_event_termination_after_z_hours
test_same_event_when_gap_less_than_z
test_new_event_after_z_hour_separation
test_consecutive_storm_hours
test_missing_timestamp_does_not_count_as_below_threshold
test_left_dataset_boundary
test_right_dataset_boundary
```

## 12. Canonical Implementation

The event definition must be implemented once under `src/definitions/` and reused by target construction, alert association, evaluation, analysis, and case studies.

## 13. Frozen Rule

```text
T = 5
Z = 6 hours
```

These values must not be modified based on model performance or final-test results.
