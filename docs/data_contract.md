# Data Contract

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Phase 0.1, Phase 0.2, and Phase 0.3 source-availability audits complete — remaining causal infrastructure pending

---

## 1. Purpose

This document defines the data that may be used by the geomagnetic storm early warning system and the temporal rules required to prevent leakage.

`MASTER_PROTOCOL_v1.3.md` is the authoritative source for methodological decisions.

The Data Contract ensures that every feature has a defined source and temporal meaning, unavailable information cannot enter the feature matrix, transformations are reproducible, and target construction remains separate from feature construction.

## 2. Primary Information Cutoff

For prediction time `t`:

```text
information_cutoff = t - 1h
maximum_feature_information_time <= information_cutoff
```

This rule applies independently to every predictor source.

## 3. OMNI Timestamp Semantics

Phase 0.1 verified that an OMNI hourly timestamp `s` marks the **start** of the represented hourly interval:

```text
period_start = s
period_end   = s + 1h
period       = [s, s + 1h)
```

An OMNI observation is eligible for prediction at `t` only when:

```text
period_end <= t - 1h
```

No feature builder may replace this rule with an unconditional row shift.

## 4. Primary Data Contract

| Variable / source | Source | Temporal meaning | Primary feature policy |
|---|---|---|---|
| Bz | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Bt | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| V | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Density | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Pressure | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Kp | OMNI / GFZ semantics | Canonical 3-hour intervals | Conservative `kp_asof()` mapping |
| AE | OMNI retrospective series | Retrospective hourly index | Excluded |
| Dst | OMNI retrospective series | Retrospective hourly index | Excluded |
| CME | SOHO/LASCO CDAW | Timestamped measurements inside a retrospectively curated candidate universe | Excluded |

The frozen primary predictor universe is:

```text
causally eligible OMNI solar-wind measurements
+
conservative causal Kp history
```

## 5. Primary Raw Feature Family

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

AE and Dst remain available in raw OMNI ingestion for audit and exploratory analysis but are not members of the primary operational feature family.

CDAW/LASCO CME data are retained as a research/audit source but are not members of the primary operational feature family.

## 6. Kp Causal Availability

Kp is a canonical 3-hour geomagnetic index. OMNI repeats each 3-hour Kp value across the three hourly rows belonging to that interval. Raw OMNI Kp uses the `Kp × 10` integer encoding and is normalized to standard Kp units.

For predictor-side Kp:

```text
kp_asof(q)
```

returns the value from the most recent canonical interval satisfying:

```text
interval_end <= q
```

The primary Kp lag family is:

```text
Kp_lag_1h(t)  = kp_asof(t - 1h)
Kp_lag_3h(t)  = kp_asof(t - 3h)
Kp_lag_6h(t)  = kp_asof(t - 6h)
Kp_lag_12h(t) = kp_asof(t - 12h)
Kp_lag_24h(t) = kp_asof(t - 24h)
```

Kp used for target/event truth remains retrospective ground truth and is not passed through predictor-side `kp_asof()`.

## 7. AE and Dst Availability

AE and Dst are retained in raw OMNI ingestion but excluded from the primary causal feature set.

The historical retrospective series cannot be demonstrated to consistently equal the value available at the historical prediction time.

```text
raw ingestion:              allowed
exploratory analysis:       allowed
primary causal feature set: excluded
```

This decision was made before model training and was not based on predictive performance.

## 8. Feature Information Rules

Raw observations, rolling statistics, persistence measures, deltas, slopes, and interactions may use only causally eligible input observations.

For every derived feature:

```text
maximum_input_information_time <= t - 1h
```

Missing timestamps must not be silently treated as measured values.

## 9. Target Information

The target is separate from the predictor matrix.

```text
X(t) = information available by t - 1h
y(t) = storm information occurring in the future target horizon
```

Future Kp may be used to construct the target and retrospective event truth, but must never enter `X(t)`.

## 10. Dataset Row Contract

Every primary dataset row must be traceable to:

```text
prediction_time
maximum_feature_information_time
target interval
target value
```

Where applicable, rows may also carry `storm_id` and other non-predictive audit metadata.

## 11. Missing Data Contract

The implementation must distinguish:

- valid measured values;
- explicit missing values;
- missing timestamps;
- unavailable source periods;
- information not yet available at prediction time.

No missing observation may be interpreted as a valid quiet physical state merely because a value is absent.

Imputation policy, if required, must be frozen before model-performance inspection and fitted only from appropriate training information.

## 12. CME Availability Contract

Phase 0.3 completed the historical-availability audit for SOHO/LASCO CDAW CME information.

The audit distinguishes:

```text
measurement causality
candidate-event causality
```

### 12.1 Measurement causality

Timestamped CDAW/LASCO height-time (`.yht`) observations provide a technically viable basis for causal kinematic reconstruction after a CME candidate has been defined.

For prediction time `t`:

```text
information_cutoff = t - 1h
```

Only measurements satisfying:

```text
measurement_time <= information_cutoff
```

are causally eligible.

The full 1996–2025 audit successfully parsed 42,422 height-time trajectories. Of these, 41,705 (98.3098%) contained at least three measurements. Among trajectories with at least three measurements, 99.8945% reached the third measurement within six hours of the first.

Measurement availability itself was therefore not the reason for excluding CME predictors.

### 12.2 Candidate-event causality

The unresolved problem is historical candidate identity.

The CDAW CME catalog is manually curated and retrospectively revised. Phase 0.3 identified 3,410 explicitly retrospective insertions in the audited event universe.

Therefore:

```text
CME observation occurred before t
```

does not establish:

```text
CME candidate was historically known at t
```

and timestamped height-time measurements do not establish that the corresponding candidate grouping was available at `t`.

Neither the investigated CDAW Version 1 nor Version 2 candidate universe provides uniform per-event historical availability semantics sufficient for the primary causal standard across 1996–2025. A Version-1/Version-2 splice is not adopted because it would introduce a catalog-regime change affecting the protected evaluation era.

### 12.3 Primary feature policy

```text
raw/research CDAW access:             allowed
CDAW source auditing:                 allowed
retrospective scientific analysis:    allowed
future separately specified extension: allowed

primary causal feature matrix:        excluded
primary feature screening:            excluded
primary model selection:              excluded
threshold optimization inputs:        excluded
```

The exclusion applies to CME counts, time-since-CME variables, final catalog kinematics, geometry, mass/energy products, and reconstructed CME kinematics.

Reconstructed kinematics remain excluded even though measurement-side reconstruction is technically feasible, because the candidate-event universe does not meet the primary historical-availability standard.

### 12.4 Non-performance-driven decision

The CME exclusion was frozen during Phase 0 before CME feature screening, model training with CME predictors, threshold optimization, or final-test inspection.

No CME variable was excluded because of predictive performance.

Detailed evidence is recorded in:

```text
docs/cme_availability.md
```

## 13. Source-Unavailability Rule

The LASCO/catalog gaps encountered during Phase 0.3 demonstrate that source unavailability must never be interpreted as zero CME activity.

Because CME is excluded from the primary feature set, no CME missingness or imputation policy is required for the primary experiment.

## 14. Required Causality / Integrity Tests

Phase 0.3 separately implemented CDAW acquisition/parser integrity tests and historical availability audits. These tests support the decision to exclude CME from the primary feature pipeline; CME future-mutation tests are therefore not required for the primary experiment.

The remaining primary pipeline must include tests covering:

```text
test_timestamp_cutoff
test_no_future_omni
test_rolling_causality
test_persistence_causality
test_delta_causality
test_slope_causality
test_target_future_only
test_missing_timestamp_handling
test_train_validation_order
test_test_isolation
test_future_mutation
```

Kp-specific causal tests are already implemented separately.

## 15. Phase 0 Verification Checklist

- [x] OMNI timestamp semantics verified
- [x] OMNI raw-field mapping verified
- [x] Kp interval semantics verified
- [x] Kp predictor-side causal mapping tested
- [x] AE/Dst historical availability audited
- [x] AE/Dst primary-feature policy frozen: excluded
- [x] CME historical availability semantics audited
- [x] CDAW retrospective catalog behavior documented
- [x] CDAW height-time measurement availability audited
- [x] CME measurement causality shown technically feasible
- [x] CME candidate-event causality limitation identified
- [x] CME primary-feature policy frozen: excluded
- [ ] Generic temporal cutoff infrastructure completed
- [ ] Event construction tested
- [ ] Alert construction tested
- [ ] Complete feature/target separation implementation
- [ ] Rolling causality tested
- [ ] Persistence causality tested
- [ ] Dynamic-feature causality tested
- [ ] Global future-mutation suite passing
- [ ] Temporal split integrity tested
- [ ] Final-test isolation tested

## 16. Completion State

```text
Data Contract specification:              COMPLETE
OMNI source verification:                 COMPLETE
Geomagnetic-index verification:           COMPLETE
CME source verification:                  COMPLETE / EXCLUDED
Primary source universe:                  FROZEN — OMNI + causal Kp
Full feature-pipeline verification:       PENDING
Full implementation/leakage verification: PENDING
```
