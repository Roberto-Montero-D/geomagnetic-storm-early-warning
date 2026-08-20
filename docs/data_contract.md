# Data Contract

**Protocol:** `MASTER_PROTOCOL_v1.2.md`  
**Status:** Phase 0.1 and Phase 0.2 verified — CME and remaining causal infrastructure pending

---

## 1. Purpose

This document defines the data that may be used by the Geomagnetic Storm Early Warning System and the temporal rules that determine whether each observation is causally available at prediction time.

It ensures that every feature has a defined source and temporal meaning, future information cannot enter the feature matrix, target construction remains separate from feature construction, retrospective information is not incorrectly treated as real-time information, and temporal assumptions can be tested automatically.

`MASTER_PROTOCOL_v1.2.md` is the authoritative source for methodological decisions.

---

## 2. Fundamental Temporal Rule

For prediction time `t`:

```text
information_cutoff = t - 1h
maximum_feature_information_time <= information_cutoff
```

For raw hourly OMNI measurements, timestamp `s` marks the **start** of the represented interval:

```text
period_start = s
period_end   = s + 1h
period       = [s, s + 1h)
```

The record is eligible only when:

```text
period_end <= t - 1h
```

Example:

```text
prediction_time = 14:00
information_cutoff = 13:00
12:00 row -> [12:00, 13:00) -> allowed
13:00 row -> [13:00, 14:00) -> not allowed
```

---

## 3. Verified Source Convention

Phase 0.1 verified that the OMNIWeb hourly timestamp is the start of the represented hourly averaging interval. Causal eligibility is determined from `period_end`, not directly from the raw timestamp.

The implementation must normalize source timestamps explicitly rather than relying on implicit dataframe shifts to repair temporal alignment.

---

## 4. Data Contract Table

| Variable | Source | Raw Temporal Meaning | Primary Feature Policy | Phase 0 Status |
|---|---|---|---|---|
| Bz | OMNI | `[s, s+1h)` | `period_end <= t-1h` | Verified |
| Bt | OMNI | `[s, s+1h)` | `period_end <= t-1h` | Verified |
| V | OMNI | `[s, s+1h)` | `period_end <= t-1h` | Verified |
| Density | OMNI | `[s, s+1h)` | `period_end <= t-1h` | Verified |
| Pressure | OMNI | `[s, s+1h)` | `period_end <= t-1h` | Verified |
| AE | OMNI | Retrospective hourly index | Excluded from primary causal feature set | Verified / excluded |
| Dst | OMNI | Retrospective hourly index | Excluded from primary causal feature set | Verified / excluded |
| Kp | OMNI / GFZ | 3-hour interval repeated across hourly OMNI rows | Canonical completed intervals via `kp_asof()` | Verified |
| CME | SOHO/LASCO catalog | Source-dependent | Pending historical-availability audit | Pending Phase 0.3 |

---

## 5. OMNI Timestamp Contract

```text
OMNI timestamp verification: VERIFIED
```

Verification record:

```text
Dataset: OMNIWeb hourly subset used by this project
Coverage: 1996-01-01 00:00 through 2025-12-31 23:00
Rows: 262,992
Timestamp source: YEAR + DOY + Hour
Raw timestamp convention: start of represented hourly interval
Physical interval: [s, s + 1h)
Normalized information time: period_end = s + 1h
Prediction-time eligibility: period_end <= t - 1h
Missing hourly timestamps: 0
Duplicate timestamps: 0
Ordering: monotonically increasing
```

The loader parses the companion `.fmt` schema, validates the expected 17-column subset, checks the actual `.lst` column count before assigning internal names, constructs the timestamp, and validates duplicate timestamps and hourly continuity.

Raw OMNI fill/sentinel values are deliberately preserved by ingestion and must be handled separately from missing timestamps.

---

## 6. Kp Timestamp and Availability Contract

### 6.1 Raw representation

OMNI stores Kp in `Kp × 10` integer encoding and repeats each 3-hour value over its three hourly rows.

```text
00:00  50
01:00  50
02:00  50
```

represents:

```text
[00:00, 03:00) -> Kp = 5.0
```

### 6.2 Canonical causal representation

Repeated hourly values are collapsed into canonical records:

```text
interval_start
interval_end
kp
```

`kp_asof(q)` returns the Kp from the most recent interval satisfying:

```text
interval_end <= q
```

### 6.3 Predictor lag semantics

```text
Kp_lag_1h(t)  = kp_asof(t - 1h)
Kp_lag_3h(t)  = kp_asof(t - 3h)
Kp_lag_6h(t)  = kp_asof(t - 6h)
Kp_lag_12h(t) = kp_asof(t - 12h)
Kp_lag_24h(t) = kp_asof(t - 24h)
```

The temporal cutoff is applied exactly once. Predictor Kp features must never be constructed directly from the repeated hourly OMNI Kp column.

This policy is a conservative historical availability approximation because the project does not reconstruct the historical GFZ nowcast stream.

### 6.4 Target/event Kp

Target and event truth use retrospective canonical Kp in standard Kp units. Predictor-side `kp_asof()` availability transformation does not apply to ground-truth construction.

---

## 7. AE and Dst Availability Contract

AE and Dst are retained during raw ingestion because ingestion must faithfully preserve the source dataset. They are excluded from the primary causal feature set because the retrospective historical products cannot be demonstrated to consistently equal the values available at the historical prediction time.

```text
raw ingestion:                  allowed
exploratory analysis:           allowed
primary causal feature matrix:  excluded
```

This exclusion was determined during Phase 0 before model training and was not based on predictive performance.

---

## 8. Feature Information Contract

Feature and target construction are conceptually independent:

```text
AVAILABLE INFORMATION -> build_features(...) -> X(t)
FUTURE INFORMATION    -> build_target(...)   -> y(t)
```

The feature builder must not depend on future target information.

---

## 9. Raw Feature Contract

Primary raw causal features:

```text
Bz
Bt
V
Density
Pressure
Kp_lag_1h
Kp_lag_3h
Kp_lag_6h
Kp_lag_12h
Kp_lag_24h
```

AE and Dst remain available in the raw ingested dataset but are not members of the primary operational feature family.

---

## 10. Rolling Feature Contract

Rolling features operate only on observations already normalized to explicit physical periods and filtered by causal eligibility.

No unconditional `.shift(1)` rule is defined.

```text
period_end = raw_timestamp + 1h
period_end <= t - 1h
```

This prevents both future leakage and unnecessary double shifting.

---

## 11. Persistence, Dynamic, and Interaction Contracts

Persistence calculations must stop at the latest causally available information. Future observations must never extend a persistence interval backward into prediction time.

Every observation used for deltas or slopes must independently satisfy the causal cutoff. Interaction features are valid only when every component variable is causally valid.

---

## 12. CME Availability Contract

CME information requires a dedicated Phase 0.3 audit. The following concepts must remain distinct:

```text
event_time
observation_time
publication_time
availability_time
```

A CME occurring before `t` does not prove that all retrospective catalog information was available at `t`.

Fundamental rule:

```text
cme_information_available_at_t == True
```

If historical availability cannot be reliably reconstructed for a CME variable, that variable must be excluded from the operational feature set. No final CME availability conclusion is made in this document before Phase 0.3.

---

## 13. Missing Data Contract

The implementation distinguishes valid observations, missing values, missing timestamps, unavailable sources, and information not yet available.

The verified OMNI source timeline from 1996–2025 contains zero missing hourly timestamps, but generic missing-timestamp behavior remains necessary for derived datasets and other sources.

A missing canonical Kp interval produces missing hourly ground-truth states and must not be treated as below threshold.

No imputation policy is frozen at this stage. Any future imputation must be causal and must never use future observations to impute past feature values.

---

## 14. Dataset Row Contract

Every model-ready row corresponds to one prediction time and must expose or permit auditing of:

```text
prediction_time
protocol_information_cutoff
maximum_feature_information_time
target_window_start
target_window_end
```

where:

```text
protocol_information_cutoff = prediction_time - 1h
```

For raw OMNI:

```text
period_end <= protocol_information_cutoff
```

Where applicable, rows may also carry `storm_id` and CME availability metadata.

---

## 15. Target Contract

Primary experiment:

```text
T = 5
H = 6h
y_event(t) = max(Kp[t+1 : t+H]) >= T
```

Future information is permitted only inside target construction. Kp used for target/event truth uses the retrospective historical Kp series in standard units.

---

## 16. Temporal Split Contract

| Split | Period |
|---|---|
| Initial Train | 1996–2016 |
| Validation 1 | 2017–2018 |
| Train 2 | 1996–2018 |
| Validation 2 | 2019–2020 |
| Train 3 | 1996–2020 |
| Validation 3 | 2021 |
| Final Test | 2022–2025 |

The historical start was extended from 2008 to 1996 during Phase 0, before model training or performance inspection, after verifying source coverage and data quality. Validation and final-test periods were not changed.

---

## 17. Required Causality Tests

Already implemented/verified for OMNI/Kp include schema validation, hourly continuity, duplicate timestamp detection, raw Kp encoding conversion, 3-hour bin structure/consistency, protocol-cutoff lag behavior, midnight/year/leap-year boundaries, future-mutation invariance, and protection from incomplete current Kp intervals.

The complete Phase 0 suite must additionally cover:

```text
test_timestamp_cutoff
test_no_future_omni
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

---

## 18. Future-Mutation Invariant

For any prediction time `t`, modifying observations strictly after the allowed information cutoff must not change `X(t)`.

```text
X_original(t) == X_future_modified(t)
```

This invariant must eventually be applied to all major feature families.

---

## 19. Phase 0 Verification Checklist

- [x] OMNI timestamp convention verified
- [x] OMNI physical interval representation documented
- [x] OMNI raw schema validated
- [x] OMNI hourly continuity verified
- [x] Kp timestamp convention verified
- [x] Kp feature availability policy verified
- [x] Kp causal alignment implemented
- [x] Kp causality tests passing
- [x] AE availability semantics audited
- [x] Dst availability semantics audited
- [x] AE/Dst primary-feature policy defined
- [ ] CME historical availability semantics verified
- [ ] Generic temporal cutoff infrastructure completed
- [ ] Event construction tested
- [ ] Alert construction tested
- [ ] Complete feature/target separation implementation
- [ ] Rolling causality tested
- [ ] Persistence causality tested
- [ ] Dynamic-feature causality tested
- [ ] CME causality tested
- [ ] Global future-mutation suite passing
- [ ] Temporal split integrity tested
- [ ] Final-test isolation tested

---

## 20. Completion Rule

```text
Data Contract specification:              COMPLETE
OMNI source verification:                 COMPLETE
Geomagnetic-index verification:           COMPLETE
CME source verification:                  PENDING
Full feature-pipeline verification:       PENDING
Full implementation/leakage verification: PENDING
```

No model training should begin before the relevant remaining Phase 0 temporal and causal requirements are verified.
