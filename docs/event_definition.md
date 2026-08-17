# Event Definition

**Protocol:** `MASTER_PROTOCOL_v1.1`  
**Primary threshold:** `T = 5`  
**Event separation parameter:** `Z = 6 hours`

## 1. Purpose

The purpose of this definition is to convert the hourly Kp time series into discrete geomagnetic storm events.

The event definition is independent of any prediction model.

A storm event is not equivalent to an individual positive hourly target.

This distinction is necessary because one physical storm can produce many consecutive hours with elevated Kp.

---

## 2. Storm Threshold

The primary storm threshold is:

```text
T = 5
```

An hour satisfies the storm condition when:

```text
Kp >= 5
```

---

## 3. Event Start

A storm event begins at the first valid hourly observation satisfying:

```text
Kp >= T
```

provided that this observation represents the beginning of a new event according to the separation rule.

Conceptually:

```text
first hour with Kp >= T
```

marks the event start.

---

## 4. Event End

An event ends when the storm condition has remained below the threshold for:

```text
Z = 6 consecutive hours
```

That is:

```text
Kp < T
```

for six consecutive valid hourly observations.

The six-hour sequence is treated as the separation period between storm activity and the next independent event.

---

## 5. Event Independence

Two periods of elevated Kp are considered separate events only when they are separated according to the full `Z = 6 hour` rule.

A short interruption below the threshold must not automatically create a new event.

This prevents a single physical storm episode from being fragmented into multiple events merely because Kp briefly falls below the threshold.

---

## 6. Missing Observations

Missing timestamps must not be treated as equivalent to:

```text
Kp < T
```

A missing observation therefore cannot automatically contribute one of the six consecutive below-threshold hours required to terminate an event.

The implementation must explicitly account for continuity of the hourly time grid.

The precise missing-data behavior must be covered by unit tests before event construction is used for evaluation.

---

## 7. Event Representation

The implementation should represent each event as a structured object or record containing, at minimum:

```text
event_id
start_time
end_time
threshold
```

Additional derived quantities may be added later without changing the event definition.

The event identifier must remain stable across:

- target construction,
- alert association,
- evaluation,
- error analysis,
- and case-study analysis.

---

## 8. Relationship to the Forecast Target

The event definition determines whether a future storm event exists.

For a prediction made at time `t`, the primary target asks whether an event begins within:

```text
t + 1h ... t + H
```

with:

```text
H = 6 hours
```

The event definition itself must be constructed independently of model predictions.

---

## 9. Required Unit Tests

Before event construction is considered valid, the implementation must test at least:

### Test 1 — Basic event start

A sequence containing a transition from:

```text
Kp < 5
```

to:

```text
Kp >= 5
```

must create an event at the first threshold-crossing hour.

### Test 2 — Six-hour separation

Two periods separated by exactly the protocol-defined termination condition must be handled according to the `Z = 6` rule.

### Test 3 — Insufficient separation

A period containing fewer than six consecutive valid below-threshold hours must not incorrectly create a new independent event.

### Test 4 — Missing timestamp

A missing hourly observation must not be interpreted as a below-threshold observation.

### Test 5 — Dataset boundary

An event that reaches the end of the available dataset must be handled deterministically and must not require unavailable future observations.

### Test 6 — Consecutive storm hours

Multiple consecutive hours satisfying `Kp >= 5` must belong to the same event unless the separation rule establishes a new event.

---

## 10. Implementation Requirement

The event definition must be implemented once and reused throughout the project.

No individual model, notebook, metric function, or evaluation script should implement its own interpretation of a storm event.

The canonical implementation belongs under:

```text
src/definitions/
```

and its behavior must be protected by automated tests.

---

## 11. Frozen Methodological Rule

The following parameters are frozen by the master protocol:

```text
T = 5
Z = 6 hours
```

They must not be modified based on model performance or test-set results.

Alternative threshold or horizon experiments must remain explicitly separated from the primary system.