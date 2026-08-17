# Data Contract

**Protocol:** `MASTER_PROTOCOL_v1.1`
**Status:** Specification complete — Phase 0 source verification pending

---

## 1. Purpose

This document defines the data that may be used by the Geomagnetic Storm Early Warning System and the temporal rules that determine whether each observation is causally available at prediction time.

The Data Contract exists to ensure that:

1. every feature has a defined source;
2. every observation has a defined temporal meaning;
3. information unavailable at prediction time cannot enter the feature matrix;
4. feature transformations are deterministic and reproducible;
5. target construction remains logically separate from feature construction;
6. retrospective information is not incorrectly treated as real-time information;
7. temporal assumptions can be verified through automated tests.

`MASTER_PROTOCOL_v1.1.md` is the authoritative source for methodological decisions. This document specifies how those decisions must be interpreted and verified during implementation.

---

## 2. Fundamental Temporal Rule

For a prediction made at time `t`, the latest permissible OMNI observation is the hourly observation whose represented period ends at:

```text
t - 1 hour
```

Therefore:

```text
latest_allowed_observation = t - 1h
```

and:

```text
maximum_feature_information_time <= t - 1h
```

No observation corresponding to the interval:

```text
(t - 1h, t]
```

may be used.

### Example

For:

```text
prediction_time = 14:00
```

the latest permissible observation period must end at:

```text
13:00
```

Therefore:

```text
Allowed:
    all valid periods ending <= 13:00

Not allowed:
    13:00-14:00
    any future period
```

The implementation of this rule depends on the verified timestamp semantics of each source.

---

## 3. Protocol Convention vs. Source Verification

The master protocol defines the operational convention required by the project.

For OMNI variables, the intended representation is:

```text
Dataset timestamp: end of represented hourly period
Period represented: [timestamp - 1h, timestamp]
```

However, this convention must not be treated as empirically verified merely because it appears in the protocol.

Phase 0 must independently verify how the actual OMNI dataset used by the project encodes its timestamps.

Therefore, two concepts must remain separate:

```text
PROTOCOL REQUIREMENT
        ↓
Observation period must end <= t - 1h

SOURCE VERIFICATION
        ↓
Determine what the actual OMNI timestamp represents
```

If the verified source convention differs from the assumed representation, the implementation must be corrected and the discrepancy documented as a Data Contract issue.

This does not constitute a performance-driven methodological change.

---

## 4. Data Contract Table

| Variable | Source      | Protocol Timestamp Convention              | Period Represented          | Availability Rule                        | Phase 0 Verification |
| -------- | ----------- | ------------------------------------------ | --------------------------- | ---------------------------------------- | -------------------- |
| Bz       | OMNI        | End of period                              | `[timestamp-1h, timestamp]` | `timestamp <= t-1h`                      | Pending              |
| Bt       | OMNI        | End of period                              | `[timestamp-1h, timestamp]` | `timestamp <= t-1h`                      | Pending              |
| V        | OMNI        | End of period                              | `[timestamp-1h, timestamp]` | `timestamp <= t-1h`                      | Pending              |
| Density  | OMNI        | End of period                              | `[timestamp-1h, timestamp]` | `timestamp <= t-1h`                      | Pending              |
| Pressure | OMNI        | End of period                              | `[timestamp-1h, timestamp]` | `timestamp <= t-1h`                      | Pending              |
| AE       | OMNI        | End of period                              | `[timestamp-1h, timestamp]` | `timestamp <= t-1h`                      | Pending              |
| Dst      | OMNI        | End of period                              | `[timestamp-1h, timestamp]` | `timestamp <= t-1h`                      | Pending              |
| Kp       | Kp index    | Exact hour / protocol-defined hourly value | Hourly value                | `timestamp <= t-1h` when used as feature | Pending              |
| CME      | CME catalog | Source-dependent                           | Event information           | Only if information was available at `t` | Pending              |

The `Phase 0 Verification` column must be updated only after the corresponding source convention has been documented and tested.

---

## 5. OMNI Timestamp Contract

### 5.1 Current status

```text
OMNI timestamp verification: PENDING
```

Before feature construction, Phase 0 must determine whether the timestamp in the exact OMNI dataset used by this project represents:

* the beginning of the averaging interval;
* the end of the averaging interval;
* an instantaneous reference time;
* or another documented convention.

No feature implementation may assume the answer before this verification is complete.

### 5.2 Required verification record

The Phase 0 audit must document:

```text
Dataset:
Source:
Source documentation:
Raw timestamp:
Physical interval represented:
Timestamp convention:
Prediction-time mapping:
Latest usable record for prediction at t:
Verification result:
```

At least one concrete timestamp example must be included.

### 5.3 Required causal mapping

After verification, the implementation must establish an unambiguous mapping:

```text
raw OMNI timestamp
        ↓
physical period represented
        ↓
period end
        ↓
prediction time t
        ↓
causal eligibility
```

An observation is eligible only if its represented information satisfies the protocol cutoff.

---

## 6. Kp Timestamp Contract

Kp serves two distinct roles in the project:

1. historical Kp may be used as a feature;
2. future Kp is used to construct the target.

These roles must remain strictly separated.

### 6.1 Kp used as a feature

Historical Kp values such as:

```text
Kp(t-1h)
Kp(t-3h)
Kp(t-6h)
Kp(t-12h)
Kp(t-24h)
```

may only be used if their information satisfies the temporal availability rule.

The exact timestamp semantics of the Kp source must therefore be verified during Phase 0.

### 6.2 Kp used for the target

Future Kp values may be used only inside target construction.

For the primary task:

```text
y_event(t) = max(Kp[t+1 : t+H]) >= T
```

with:

```text
T = 5
H = 6 hours
```

Future Kp information must never enter `X(t)`.

---

## 7. Feature Information Contract

Feature construction and target construction must be implemented as conceptually independent operations.

```text
PAST / AVAILABLE INFORMATION
            ↓
     build_features(...)
            ↓
           X(t)


FUTURE INFORMATION
            ↓
      build_target(...)
            ↓
           y(t)
```

The feature builder must not depend on future target information.

The target builder must not modify or influence feature construction.

---

## 8. Raw Feature Contract

The primary raw feature family defined by the master protocol includes:

```text
Bz
Bt
V
Density
Pressure
AE
Dst
Kp(t-1h)
Kp(t-3h)
Kp(t-6h)
Kp(t-12h)
Kp(t-24h)
```

Every raw feature must satisfy the causal cutoff before being included in the feature matrix.

---

## 9. Rolling Feature Contract

Rolling features may only use observations that independently satisfy the temporal cutoff.

The protocol includes rolling statistics over windows such as:

```text
3h
6h
12h
24h
```

including quantities such as:

```text
rolling mean
rolling minimum
rolling standard deviation
```

Before implementing the exact rolling operation, the OMNI timestamp convention must be verified.

In particular, the project must determine whether an additional shift is required after applying the temporal cutoff.

The implementation must avoid both:

```text
future leakage
```

and:

```text
unnecessary double shifting
```

The correct behavior must be established through timestamp analysis and unit tests rather than assumption.

---

## 10. Persistence Feature Contract

Persistence features measure the duration of previously observed physical conditions.

Examples defined by the protocol include:

```text
Bz_negative_less_than_-5_duration
Bz_negative_less_than_-10_duration
Bz_negative_less_than_-15_duration

V_high_greater_than_500_duration
V_high_greater_than_600_duration
```

Persistence calculations must stop at the latest causally available observation.

Future observations must never extend a persistence interval backward into prediction time `t`.

---

## 11. Dynamic Feature Contract

Dynamic features include changes and trends such as:

```text
delta_Bz_1h
delta_Bz_3h
delta_V_1h
delta_V_3h
slope_Bz_3h
slope_V_3h
```

Every observation used to calculate a difference or slope must independently satisfy the causal cutoff.

For example, a slope evaluated for prediction time `t` must not use a point whose represented period extends beyond the latest allowed observation.

---

## 12. Interaction Feature Contract

Interaction features may combine causally valid variables.

Examples defined by the protocol include:

```text
Bz_neg_multiply_V
Bz_neg_multiply_Density
Pressure_multiply_V
```

An interaction feature is causally valid only when **all** of its component variables are causally valid.

A valid historical variable combined with an unavailable variable produces an invalid feature.

---

## 13. CME Availability Contract

CME information requires stricter treatment than ordinary retrospective observations.

The following concepts must remain distinct:

```text
event_time
observation_time
publication_time
availability_time
```

The occurrence of a CME before prediction time `t` does **not** prove that all catalog information associated with that CME was available at `t`.

The fundamental CME rule is:

```text
cme_information_available_at_t == True
```

Only CME information satisfying this rule may enter the feature matrix.

### 13.1 Potential CME features

The master protocol permits features such as:

```text
hours_since_cme_observation
hours_until_cme_eta
cme_speed_if_available
cme_energy_if_available
```

and derived temporal CME features such as:

```text
hours_since_last_CME
days_since_last_CME

CME_count_last_24h
CME_count_last_48h
CME_count_last_72h

max_CME_speed_last_24h
max_CME_speed_last_48h
max_CME_speed_last_72h

hours_until_CME_eta
```

These features are permitted only when the underlying information was operationally available at prediction time.

### 13.2 CME exclusion rule

If historical real-time availability cannot be reliably reconstructed for a CME variable, that variable must not be treated as operationally available merely because it exists in a retrospective catalog.

The issue must be documented.

Features whose causal availability cannot be demonstrated must be excluded from the operational feature set.

Such exclusion is a Data Contract correction, not a model-performance decision.

---

## 14. Missing Data Contract

Missing observations must not be silently interpreted as physical measurements.

The implementation must distinguish between:

```text
valid observation
missing value
missing timestamp
source unavailable
information not yet available
```

These states are not equivalent.

### 14.1 Missing timestamps

Temporal continuity must be explicitly checked.

For example:

```text
12:00  Kp = 4
13:00  MISSING
14:00  Kp = 4
```

must not automatically be interpreted as three consecutive valid below-threshold hourly observations.

Missing timestamps must therefore not manufacture artificial:

* event termination;
* persistence duration;
* rolling continuity;
* or alert continuity.

### 14.2 Imputation

No imputation policy is defined by this Data Contract at this stage.

Any required imputation strategy must be documented during dataset construction and must respect temporal causality.

Future observations must never be used to impute past feature values.

---

## 15. Dataset Row Contract

Every model-ready dataset row must correspond to a unique prediction time:

```text
t
```

For every row, the implementation must be able to establish:

```text
prediction_time
latest_allowed_observation
maximum_feature_information_time
target_window_start
target_window_end
```

and, where applicable:

```text
storm_id
CME availability information
```

The core invariant is:

```text
maximum_feature_information_time <= prediction_time - 1h
```

for feature information governed by the protocol cutoff.

---

## 16. Target Contract

For the primary experiment:

```text
T = 5
H = 6 hours
```

and:

```text
y_event(t) = max(Kp[t+1 : t+H]) >= T
```

The target uses future information by definition.

This is not leakage because future information is used exclusively to define the outcome being predicted.

Leakage occurs if any of that future information enters the feature matrix or influences feature construction.

---

## 17. Temporal Split Contract

Dataset construction must preserve chronological order.

The frozen primary splits are:

| Split         | Period    |
| ------------- | --------- |
| Initial Train | 2008–2016 |
| Validation 1  | 2017–2018 |
| Train 2       | 2008–2018 |
| Validation 2  | 2019–2020 |
| Train 3       | 2008–2020 |
| Validation 3  | 2021      |
| Final Test    | 2022–2025 |

No future validation or test information may influence an earlier training period.

The final test period is protected and must not be used during feature selection, model selection, balancing selection, hyperparameter optimization, or threshold selection.

---

## 18. Required Causality Tests

Before model development, the implementation must include automated tests covering at least:

```text
test_timestamp_cutoff
test_no_future_omni
test_no_future_kp
test_rolling_causality
test_persistence_causality
test_delta_causality
test_slope_causality
test_cme_availability
test_target_future_only
test_missing_timestamp_handling
test_train_validation_order
test_test_isolation
test_future_mutation
```

These tests are part of Phase 0.

---

## 19. Future-Mutation Test

The project must include a strong causal-invariance test.

Procedure:

1. construct features for prediction time `t`;
2. store the resulting feature vector;
3. modify observations occurring strictly after the allowed information cutoff;
4. reconstruct the features for the same prediction time `t`;
5. compare both feature vectors.

Expected result:

```text
X_original(t) == X_future_modified(t)
```

If changing future observations changes the feature vector at `t`, the feature pipeline contains temporal leakage.

This test should eventually be applied to all major feature families.

---

## 20. Phase 0 Verification Checklist

Before this Data Contract is considered operationally verified:

* [ ] OMNI timestamp convention verified
* [ ] OMNI physical interval representation documented
* [ ] Kp timestamp convention verified
* [ ] Kp feature availability verified
* [ ] CME source identified
* [ ] CME historical availability semantics verified
* [ ] Temporal cutoff implemented
* [ ] Missing timestamp behavior defined and tested
* [ ] Event construction tested
* [ ] Alert construction tested
* [ ] Feature/target separation implemented
* [ ] Rolling causality tested
* [ ] Persistence causality tested
* [ ] Dynamic-feature causality tested
* [ ] CME causality tested
* [ ] Future-mutation test passing
* [ ] Temporal split integrity tested
* [ ] Final-test isolation tested

---

## 21. Completion Rule

The Data Contract becomes operationally verified only when the Phase 0 source audits and causal tests demonstrate that:

```text
X(t)
```

contains only information that could legitimately have been available under the frozen protocol at prediction time `t`.

Until then:

```text
Data Contract specification: COMPLETE
Source verification:         PENDING
Implementation verification: PENDING
```

No model training should begin before the relevant Phase 0 temporal and causal requirements have been verified.
