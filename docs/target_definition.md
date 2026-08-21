# Canonical Target Definition

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Implemented, tested, and frozen

## 1. Purpose

This document defines the canonical retrospective supervised-learning target
implemented in Phase 0.8.

The primary target is historically named:

```text
y_event
```

but its mathematical definition is a **future storm-condition target**, not an
event-onset-only target.

The canonical implementation is:

```text
src/targets/event_window.py
```

## 2. Frozen Primary Target

For prediction time `t`:

```text
y_event(t) = max(Kp[t+1 : t+H]) >= T
```

with:

```text
T = 5
H = 6 hours
```

The exact hourly target window is:

```text
t + 1h
t + 2h
...
t + H
```

equivalently:

```text
(t, t + H]
```

Therefore:

```text
t is excluded
t + H is included
```

## 3. Semantic Clarification

Despite the historical variable name `y_event`, the target asks:

```text
Will geomagnetic storm conditions (Kp >= T) occur at any time
during the next H hours?
```

It does **not** ask only:

```text
Will a new canonical storm event start during the next H hours?
```

Consequently, if a storm has already begun at prediction time `t` but
storm-level Kp remains present at one or more future hourly states inside
`(t, t+H]`, the target is positive.

This clarification follows directly from the frozen mathematical target
definition and does not change the protocol based on model results.

Canonical storm-event objects remain necessary for event-level evaluation,
lead time, false-alarm classification, censoring, and operational alert
metrics. They are not substituted for the supervised target definition.

## 4. Ground-Truth Source

The target uses retrospective canonical Kp ground truth.

Canonical Kp is represented as 3-hour intervals:

```text
[interval_start, interval_end)
```

with:

```text
interval_end - interval_start = 3h
```

For retrospective target construction only, each canonical 3-hour Kp value is
expanded onto the three hourly states represented by that interval.

For example:

```text
[03:00, 06:00), Kp = 5
```

provides retrospective hourly ground truth at:

```text
03:00
04:00
05:00
```

Temporal gaps between canonical Kp intervals remain missing.

No forward filling across an absent canonical interval is permitted.

## 5. Separation from Predictor Availability

Target construction and predictor construction intentionally use different
temporal semantics.

Predictor-side Kp features use:

```text
kp_asof(query_time)
```

with the conservative availability rule:

```text
interval_end <= query_time
```

The retrospective target does not use `kp_asof()`.

The target is allowed to inspect the future retrospective ground truth because
it is the supervised label. Predictor features remain restricted to
information causally available at prediction time.

Therefore:

```text
future Kp may define y
future Kp may never define X
```

This separation is mandatory.

## 6. Missing Future Ground Truth

A negative target requires complete knowledge of the entire future horizon.

For the H hourly states in `(t, t+H]`:

```text
if any observed future Kp >= T:
    target = 1

elif all H future hourly Kp states are observed
and every observed Kp < T:
    target = 0

else:
    target = NaN
```

This means a known positive is sufficient even when another hour in the same
future window is missing, because the existential target condition has already
been established.

However, absence of an observed positive is not enough to assign zero when
future ground truth is incomplete.

## 7. Right-Edge Semantics

Prediction times near the end of the retrospective dataset may not have a
complete H-hour future horizon.

Such rows must not be silently labeled negative.

If the available part of the horizon already contains a known Kp value
satisfying:

```text
Kp >= T
```

the target is known positive.

Otherwise, if one or more required future hourly states extend beyond the
available retrospective ground truth:

```text
target = NaN
```

Rows with unknown targets must be excluded from supervised model fitting and
target-based scoring, while their existence may remain visible in dataset
audits.

## 8. Missing Internal Kp States

Missing Kp ground truth inside an otherwise covered future window is treated
the same way as right-edge incompleteness.

A missing canonical interval is not interpreted as quiet geomagnetic
conditions.

Therefore:

```text
missing Kp != Kp < T
```

If a positive state is observed elsewhere in the window, the target remains
known positive.

If no positive is observed and one or more future states are missing, the
target remains unknown (`NaN`).

## 9. Target Audit Fields

The canonical target builder may return an audit frame containing:

```text
future_window_start
future_window_end
expected_future_hours
observed_future_hours
missing_future_hours
positive_future_hours
target_status
```

Frozen `target_status` values are:

```text
positive
negative
unknown
```

The audit exists to make incomplete future truth explicit rather than hiding
it inside the target series.

## 10. Temporal Invariance Requirements

The target implementation must satisfy the following invariants.

### Past-ground-truth mutation invariance

Changing retrospective Kp states that are entirely before the target window
must not alter the target.

### Beyond-horizon mutation invariance

Changing retrospective Kp states strictly after `t + H` must not alter the
target.

### Boundary invariance

The target must always use exactly:

```text
(t, t+H]
```

and never accidentally include `t` or extend beyond `t+H`.

## 11. Input Validation

Canonical target construction rejects malformed temporal inputs, including:

```text
duplicate prediction timestamps
non-monotonic prediction timestamps
prediction timestamps not aligned to whole hours
overlapping canonical Kp intervals
canonical Kp intervals not exactly 3 hours long
non-finite non-missing Kp values
invalid target threshold
invalid target horizon
```

These checks protect the target definition from silent temporal corruption.

## 12. Canonical Implementation and Tests

Implementation:

```text
src/targets/event_window.py
```

Package export:

```text
src/targets/__init__.py
```

Primary tests:

```text
tests/test_targets_event_window.py
```

The tests explicitly cover:

```text
future storm-condition positives
already-active storms
left boundary exclusion
right boundary inclusion
states beyond the horizon
complete negative horizons
missing future Kp
known positives with other missing states
incomplete right-edge horizons
missing canonical intervals
past mutation invariance
beyond-horizon mutation invariance
malformed intervals and parameters
```

Target construction elsewhere in the repository must reuse the canonical
implementation rather than reconstructing equivalent labels independently.

## 13. Frozen Phase 0.8 Rule

The primary supervised target is frozen as:

```text
T = 5
H = 6h
window = (t, t+H]
condition = any retrospective hourly Kp >= T
```

with conservative unknown-label handling:

```text
known positive                         -> 1
complete known all-negative horizon    -> 0
otherwise                              -> NaN
```

The historical name `y_event` does not redefine the mathematical semantics as
event-onset-only.

Any future alternative target, including a fixed-horizon target or a
storm-onset-only target, must be treated as a separately named protocol
variant and must not silently replace the primary target.
