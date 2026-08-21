# Geomagnetic Storm Event Definition

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Implemented, tested, and frozen

## 1. Purpose

This document defines how retrospective canonical Kp is converted into discrete geomagnetic storm events. The definition is deterministic and independent of model predictions, probability thresholds, alert episodes, model selection, and model performance.

## 2. Primary Parameters

```text
T = 5
Z = 6 hours
```

## 3. Event Start

An event starts at the start of the first canonical Kp interval satisfying:

```text
Kp >= T
```

OMNI's three repeated hourly rows do not represent three independent Kp observations.

## 4. Event Continuation

Once started, an event remains active until the complete termination condition is satisfied. If Kp returns to or exceeds T before Z consecutive valid below-threshold hourly ground-truth states occur, the same event continues.

## 5. Event Termination

Canonical 3-hour Kp intervals are represented on the hourly ground-truth timeline. With complete aligned data, `Z = 6h` corresponds to two complete consecutive 3-hour Kp intervals below threshold.

The storm end is the last active storm hour before the terminating quiet sequence.

## 6. Event Separation

```text
fewer than 6 valid consecutive hours below T -> same event
6 valid consecutive hours below T           -> previous event terminated
subsequent Kp >= T                           -> new event
```

## 7. Missing Observations

Missing timestamps, Kp values, or canonical Kp intervals are not interpreted as `Kp < T` and do not contribute to the termination sequence. Missing ground truth resets the observed consecutive-quiet run.

Malformed non-missing Kp values are integrity errors and must raise rather than being silently converted to missing ground truth.

## 8. Dataset Boundaries

Dataset-boundary events are explicitly censored. The implementation never invents observations outside the available retrospective Kp record.

```text
complete        = both event boundaries are observed
left_censored   = dataset begins while storm conditions are already active
right_censored  = dataset ends before event termination is observed
both_censored   = both conditions apply
```

For a left-censored event, `start_time` is the first observed storm timestamp and is not interpreted as the physical onset.

For a right-censored event:

```text
end_time = NaT
```

The final available timestamp is not substituted for an unobserved physical storm end.

## 9. Canonical Event Representation

Each event contains:

```text
event_id
start_time
end_time
threshold
peak_kp
boundary_status
```

`boundary_status` is one of `complete`, `left_censored`, `right_censored`, or `both_censored`.

## 10. Relationship to Forecast Target

Event identification uses retrospective canonical Kp. Predictor-side `kp_asof()` is not used for event or target ground truth.

## 11. Tests

The canonical test suite covers event starts, interval alignment, repeated hourly Kp, termination, event separation, missing data, dataset censoring, and malformed Kp integrity failures.

## 12. Canonical Implementation

Implemented in:

```text
src/definitions/events.py
```

Tested in:

```text
tests/test_events.py
```

Independent reconstruction of storm boundaries elsewhere in the project is prohibited.

## 13. Frozen Rule

```text
T = 5
Z = 6 hours
```

These parameters and censoring semantics must not be changed based on model performance or final-test results.
