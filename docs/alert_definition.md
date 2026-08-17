# Alert Definition

**Protocol:** `MASTER_PROTOCOL_v1.1`  
**Primary forecast horizon:** `H = 6 hours`  
**Alert persistence:** `C = 3 hours`  
**Maximum FAR/day:** `0.2`

## 1. Purpose

The alert definition converts model probabilities into operational warning episodes.

The purpose is to evaluate whether the system can issue useful warnings for discrete geomagnetic storm events.

An individual positive hourly prediction is not automatically considered an independent operational alert.

---

## 2. Probability Threshold

At prediction time `t`, the model produces:

```text
P(event within H hours)
```

An alert is issued when:

```text
P(event within H hours) >= tau
```

where `tau` is the probability threshold selected according to the master protocol.

The primary threshold is selected using out-of-fold predictions and the protocol-defined False Alarm Rate constraint.

The final test set must not be used to select `tau`.

---

## 3. Primary False Alarm Constraint

The primary operational threshold is the minimum threshold satisfying:

```text
FAR/day <= 0.2
```

The threshold-selection procedure must be applied to the appropriate development / out-of-fold predictions defined by the master protocol.

The final test period is used only after the threshold has been frozen.

---

## 4. Alert Persistence

The primary alert persistence parameter is:

```text
C = 3 hours
```

Consecutive alert-producing predictions are grouped into an alert episode according to this persistence rule.

The purpose is to avoid treating every hourly positive prediction as a separate operational warning.

---

## 5. Alert Episode

The canonical alert object should contain, at minimum:

```text
alert_episode_id
first_alert_time
last_alert_time
classification
associated_event_id
lead_time
```

The exact representation may evolve during implementation, but the underlying definition must remain consistent with the frozen protocol.

---

## 6. Alert-to-Event Association

An alert episode must be evaluated against the event definition rather than against individual positive Kp hours.

Each alert episode should be classified according to its relationship with the storm-event timeline.

The primary categories are:

```text
Early Detection
Late Detection
False Alarm
```

The classification must be deterministic and implemented independently from the model itself.

---

## 7. Early Detection

An alert is an **Early Detection** when the alert episode provides a warning before the associated storm event begins, according to the protocol-defined temporal relationship.

The lead time is measured from the first qualifying alert to the event start.

Conceptually:

```text
first alert
     ↓
     lead time
     ↓
event start
```

---

## 8. Late Detection

An alert is a **Late Detection** when the system identifies an event only after the event has already begun, while still satisfying the protocol's association rules.

Late detections must be reported separately from genuine early warnings.

They must not be silently counted as equivalent to successful early warnings.

---

## 9. False Alarm

An alert episode is a **False Alarm** when it cannot be associated with a valid storm event under the protocol-defined association rules.

False alarms contribute to operational false-alarm statistics.

Repeated alert hours within the same alert episode must not be counted as independent false alarms.

---

## 10. Event Recall

The primary event-level recall is based on the number of storm events that receive a qualifying detection.

Conceptually:

```text
Event Recall =
detected storm events
---------------------
all storm events
```

The implementation must define detection at the event level rather than by counting hourly target labels.

---

## 11. False Alarm Rate per Day

The primary operational false-alarm constraint is:

```text
FAR/day <= 0.2
```

False alarms must be counted using alert episodes rather than individual hourly predictions.

The denominator and exact calculation must remain consistent across development, validation, OOF, and final-test evaluation.

---

## 12. Lead Time

For an early detection, lead time is measured between:

```text
first qualifying alert
```

and:

```text
storm event start
```

Positive lead time indicates that the warning preceded the event.

Lead-time distributions should be reported in addition to aggregate detection metrics.

---

## 13. Alert Episode Integrity

The implementation must guarantee that:

- consecutive alert hours are not double-counted;
- one storm event cannot be artificially counted as multiple detections;
- one alert episode cannot be counted repeatedly;
- false alarms are counted at the episode level;
- alert/event associations are deterministic.

---

## 14. Required Unit Tests

The implementation must include tests for at least:

### Test 1 — Consecutive alerts

Multiple consecutive alert hours must form one alert episode according to `C = 3`.

### Test 2 — Isolated alert

An isolated alert must remain a valid episode and must not be duplicated.

### Test 3 — Early detection

An alert occurring before an event begins must be classified as an early detection when association criteria are satisfied.

### Test 4 — Late detection

An alert occurring after event onset must be classified separately from an early detection.

### Test 5 — False alarm

An alert with no valid associated event must be classified as a false alarm.

### Test 6 — Multiple alerts near one event

Multiple alert episodes associated with the same storm must be handled according to the protocol without artificially increasing event recall.

### Test 7 — Dataset boundaries

Alerts and events near the beginning or end of the evaluation period must be handled without requiring unavailable information.

---

## 15. Threshold Selection and Test Protection

The threshold `tau` must be selected without access to the final test outcomes.

The correct conceptual sequence is:

```text
Development data
      ↓
Model training
      ↓
Out-of-fold predictions
      ↓
Threshold selection
      ↓
Frozen threshold
      ↓
Final test
```

The final test must not be used to choose a more favorable threshold after observing test performance.

---

## 16. Frozen Primary Parameters

The primary alert definition uses:

```text
H = 6 hours
C = 3 hours
maximum FAR/day = 0.2
```

These values are defined by `MASTER_PROTOCOL_v1.1`.

Alternative horizon/threshold experiments must be treated as separate generalization experiments and must not silently redefine the primary system.