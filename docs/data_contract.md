# Data Contract

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Status:** Phase 0 complete — primary causal contract frozen

---

## 1. Purpose

This document defines the data that may be used by the geomagnetic storm early warning system and the temporal rules required to prevent leakage.

`MASTER_PROTOCOL_v1.3.md` is the authoritative source for methodological decisions.

The frozen primary predictor universe is:

```text
causally eligible OMNI solar-wind measurements
+
conservative causal Kp history
```

## 2. Primary Information Cutoff

For prediction time `t`:

```text
information_cutoff = t - 1h
maximum_feature_information_time <= information_cutoff
```

This rule applies independently to every predictor source.

## 3. OMNI Timestamp Semantics

An OMNI hourly timestamp `s` marks the start of the represented hourly interval:

```text
period_start = s
period_end   = s + 1h
period       = [s, s + 1h)
```

An OMNI observation is eligible only when:

```text
period_end <= t - 1h
```

No feature builder may replace this rule with an unconditional row shift.

## 4. Primary Data Contract

| Variable / source | Temporal meaning | Primary feature policy |
|---|---|---|
| Bz | OMNI `[s,s+1h)` | Eligible if `period_end <= t-1h` |
| Bt | OMNI `[s,s+1h)` | Eligible if `period_end <= t-1h` |
| V | OMNI `[s,s+1h)` | Eligible if `period_end <= t-1h` |
| Density | OMNI `[s,s+1h)` | Eligible if `period_end <= t-1h` |
| Pressure | OMNI `[s,s+1h)` | Eligible if `period_end <= t-1h` |
| Kp | Canonical 3-hour intervals | Conservative `kp_asof()` |
| AE | Retrospective hourly index | Excluded |
| Dst | Retrospective hourly index | Excluded |
| CME | CDAW/LASCO retrospective candidate universe | Excluded |

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

The complete frozen primary feature manifest contains 93 features across raw, rolling, persistence, dynamics, and interaction families.

## 6. Kp Causal Availability

Predictor-side Kp uses:

```text
kp_asof(q)
```

defined as the most recent canonical Kp interval satisfying:

```text
interval_end <= q
```

Primary Kp lags are:

```text
1h, 3h, 6h, 12h, 24h
```

Retrospective Kp used for target/event truth is separate from predictor-side Kp availability.

## 7. AE, Dst, and CME Policy

AE and Dst remain available in raw ingestion/audit contexts but are excluded from the primary causal feature set because historical retrospective values cannot be demonstrated to consistently equal values available at historical prediction time.

CDAW/LASCO CME information is retained for research and reproducibility but excluded from the primary causal experiment because uniform historical candidate-event availability semantics could not be established across the study period.

These exclusions were frozen before model training and were not performance-driven.

## 8. Feature Information Rules

For every derived feature:

```text
maximum_input_information_time <= t - 1h
```

The frozen feature layer does not permit:

```text
forward fill
backward fill
nearest-row substitution
implicit interpolation
fallback to an older raw timestamp
```

Missing timestamps are not silently treated as measured physical states.

## 9. Target Information

The canonical primary target is:

```text
y_event(t) = max(Kp[t+1:t+H]) >= T

T = 5
H = 6h
window = (t, t+H]
```

Its semantics are future storm-condition presence, not event-onset-only prediction.

Target truth is retrospective and separate from `X(t)`.

Missing future truth follows:

```text
any known positive                    -> 1
complete all-negative future horizon  -> 0
otherwise                             -> NaN
```

## 10. Dataset Row Contract

Every primary dataset row must be traceable to:

```text
prediction_time
maximum_feature_information_time
target interval
target value
```

Rows may additionally carry non-predictive audit metadata such as event identifiers or target-status fields.

Audit metadata must not silently enter the predictor matrix.

## 11. Missing Data Contract

The implementation distinguishes:

- valid measured values;
- explicit missing values;
- missing timestamps;
- unavailable source periods;
- information not yet available at prediction time.

No missing observation is interpreted as a valid quiet physical state merely because a value is absent.

Any later model-side imputation policy must be frozen before performance-driven selection and fitted only from appropriate training information.

## 12. Phase 0 Verification Checklist

- [x] OMNI timestamp semantics verified
- [x] OMNI raw-field mapping verified
- [x] Kp interval semantics verified
- [x] Kp predictor-side causal mapping tested
- [x] AE/Dst historical availability audited
- [x] AE/Dst primary-feature exclusion frozen
- [x] CME historical availability audited
- [x] CME primary-feature exclusion frozen
- [x] Generic temporal cutoff infrastructure completed
- [x] Event construction implemented and tested
- [x] Alert construction implemented and tested
- [x] 93-feature causal pipeline implemented and frozen
- [x] Rolling causality tested
- [x] Persistence causality tested
- [x] Dynamic-feature causality tested
- [x] Interaction-feature causality tested
- [x] Canonical target construction implemented and tested
- [x] Feature/target separation tested
- [x] Global future-mutation suite passing
- [x] Temporal split-boundary integrity tested
- [x] Phase 0 temporal-integrity suite passing

## 13. Completion State

```text
Data Contract specification:              COMPLETE
OMNI source verification:                 COMPLETE
Geomagnetic-index verification:           COMPLETE
CME source verification:                  COMPLETE / EXCLUDED
Primary source universe:                  FROZEN — OMNI + causal Kp
Full causal feature-pipeline verification: COMPLETE
Target construction verification:         COMPLETE
Full Phase 0 leakage verification:         COMPLETE
Phase 0:                                   COMPLETE
```

The protected 2022–2025 Final Test has not been used for model evaluation. Phase 0 split-boundary tests verify temporal semantics only and do not constitute inspection of the real Final Test dataset.

Phase 1 may now construct the canonical dataset under this frozen contract.
