# Data Contract

**Protocol:** `MASTER_PROTOCOL_v1.1`  
**Status:** Specification in progress — Phase 0 verification required

## 1. Purpose

This document defines the data that may be used by the geomagnetic storm early warning system.

The Data Contract exists to ensure that:

1. every feature has a defined source;
2. every observation has a defined temporal meaning;
3. information unavailable at prediction time cannot enter the feature matrix;
4. transformations are deterministic and reproducible;
5. target construction remains separate from feature construction.

The master protocol takes precedence over this document.

---

## 2. Primary Information Cutoff

For a prediction made at time `t`, the latest permissible observation is:

```text
t - 1 hour
```

Therefore:

```text
maximum_feature_information_time <= t - 1h
```

This rule applies to all observational inputs.

The exact implementation depends on the verified timestamp convention of each source.

---

## 3. Timestamp Semantics

### 3.1 OMNI

The OMNI timestamp convention has **not yet been formally verified**.

Before feature construction, Phase 0 must establish whether an OMNI timestamp represents:

- the beginning of an averaging interval,
- the end of an averaging interval,
- an instantaneous observation,
- or another convention.

No implementation should assume the answer.

The verified convention must be recorded together with:

- source documentation,
- an example timestamp,
- the physical period represented,
- the corresponding prediction-time cutoff,
- and a unit test.

### 3.2 Kp

The Kp timestamp convention must also be explicitly verified before target construction.

The implementation must document what physical interval each Kp timestamp represents and how that interval maps to the hourly prediction grid.

### 3.3 CME information

CME event time and CME information availability time are distinct concepts.

The occurrence of a CME does not imply that all associated catalog information was available at that same time.

---

## 4. Required Variables

The primary Data Contract includes the following information categories.

| Variable | Source | Role | Availability requirement |
|---|---|---|---|
| Bz | OMNI | Solar-wind / IMF state | Available by `t - 1h` |
| Bt | OMNI | Magnetic-field magnitude | Available by `t - 1h` |
| V | OMNI | Solar-wind speed | Available by `t - 1h` |
| Density | OMNI | Solar-wind plasma state | Available by `t - 1h` |
| Pressure | OMNI | Solar-wind dynamic pressure | Available by `t - 1h` |
| AE | Geomagnetic index source | Geomagnetic state | Available according to source latency |
| Dst | Geomagnetic index source | Geomagnetic state | Available according to source latency |
| Kp | Geomagnetic index source | Target / persistence information | Must respect temporal cutoff when used as a feature |
| CME | CME catalog | External event information | Only if real-time availability is established |

The exact raw field names will be fixed during Phase 0 and dataset construction.

---

## 5. Feature Information Rules

### 5.1 Raw observations

A raw observation may be used only if the physical information represented by that observation was available by:

```text
t - 1h
```

### 5.2 Lagged variables

Lagged variables are allowed when the lag refers to information that was already available at prediction time.

For example:

```text
Kp(t-1)
```

may be valid if the corresponding Kp observation was available under the verified timestamp convention.

### 5.3 Rolling statistics

Rolling statistics must be computed exclusively from observations satisfying the temporal cutoff.

A rolling window must never accidentally include the current prediction period or any future period.

The implementation must have automated tests for this condition.

### 5.4 Derived variables

Derived variables such as:

- differences,
- slopes,
- rolling means,
- rolling minima/maxima,
- persistence measures,
- interaction terms,

are permitted only when every input observation satisfies the same causal information rule.

---

## 6. Target Information

The target is fundamentally different from the feature matrix.

The target may use future Kp observations because the target represents a future forecasting outcome.

Conceptually:

```text
X(t) = information available by t - 1h

y(t) = event information occurring within the future horizon
```

Future information used to construct `y(t)` must never be reused as part of `X(t)`.

The target implementation will be specified separately from the feature builder.

---

## 7. CME Availability Contract

CME information requires an explicit real-time availability rule.

The following concepts must remain separate:

```text
event_time
observation_time
publication_time
availability_time
```

A CME feature may only be used when:

```text
availability_time <= prediction_time
```

or according to the exact operational availability rule established by the protocol.

A retrospective catalog entry with a historical event time is insufficient evidence that the information was available to a forecaster at that time.

Phase 0 must therefore determine whether the selected CME source supports reconstruction of historical real-time availability.

If reliable availability cannot be established, the corresponding CME feature must not be treated as operationally available merely because it exists in a retrospective catalog.

---

## 8. Missing Data

Missing observations must not be silently interpreted as valid physical states.

The implementation must distinguish between:

- a measured value,
- an explicitly missing value,
- a timestamp gap,
- and an observation that is unavailable because of source latency.

In particular, missing hourly observations must not be used to manufacture artificial persistence for event or alert definitions.

Missing-data handling will be finalized during dataset construction after the source audits are complete.

---

## 9. Temporal Integrity Requirements

Every dataset row must be traceable to a prediction time:

```text
t
```

and must satisfy:

```text
maximum_feature_information_time <= t - 1h
```

The dataset construction pipeline must make it possible to audit:

- prediction timestamp,
- latest feature observation,
- source timestamps,
- target interval,
- target event identifier where applicable.

---

## 10. Required Validation Tests

Before model development, the following tests must exist:

```text
test_timestamp_cutoff
test_no_future_omni_information
test_no_future_kp_information
test_rolling_feature_causality
test_lag_feature_causality
test_target_uses_future_only
test_cme_availability
test_missing_timestamp_handling
test_future_data_mutation
```

The exact test implementation belongs to Phase 0.

---

## 11. Phase 0 Completion Criteria

The Data Contract will be considered implementation-ready only when:

- OMNI timestamp semantics have been verified;
- Kp timestamp semantics have been verified;
- CME availability semantics have been verified or explicitly excluded;
- all required raw fields have been mapped;
- the prediction-time cutoff is executable;
- causal feature tests pass;
- target construction is isolated from feature construction;
- missing timestamps have an explicit policy.

Until then, the Data Contract remains a specification rather than an executable guarantee.