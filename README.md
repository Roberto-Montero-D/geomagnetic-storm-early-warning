# Geomagnetic Storm Early Warning System

**Status:** Protocol Frozen — Phase 0 Complete; Phase 1 Next  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Primary Horizon:** 6 hours  
**Primary Storm Threshold:** Kp ≥ 5

## 1. Project Overview

This repository implements a scientifically controlled early-warning system for geomagnetic storms.

The operational question is:

> Given only information that would have been available at prediction time `t`, can the system issue a reliable warning that geomagnetic storm conditions will occur within the next `H` hours?

Phase 0 is complete. The primary predictor universe is frozen to:

```text
causally eligible OMNI solar-wind measurements
+
conservative causal Kp history
```

AE, Dst, and CDAW/LASCO CME-derived predictors are excluded from the primary causal feature matrix under the Phase 0 source-availability audits.

## 2. Frozen Primary Configuration

| Parameter | Primary value |
|---|---:|
| Storm threshold `T` | 5 |
| Forecast horizon `H` | 6 h |
| Event separation / termination `Z` | 6 h |
| Alert episode gap `C` | 3 h |
| Maximum FAR/day | 0.2 |
| Information cutoff | `t - 1h` |
| Historical development start | 1996 |
| Protected Final Test | 2022–2025 |
| Primary causal features | 93 |

The protected Final Test remains single-use and must not influence development decisions.

## 3. Phase 0 Completion

Completed Phase 0 infrastructure includes:

- verified OMNI hourly timestamp semantics;
- canonical 3-hour Kp intervals and conservative predictor-side `kp_asof`;
- AE/Dst historical-availability audit and primary exclusion;
- full CDAW/LASCO CME availability audit and primary exclusion;
- canonical temporal cutoff infrastructure;
- canonical storm-event construction;
- canonical alert-episode construction and event association;
- frozen 93-feature causal feature pipeline;
- canonical future storm-condition target construction;
- project-wide leakage and temporal-integrity tests.

The canonical target is:

```text
y_event(t) = max(Kp[t+1:t+H]) >= T

T = 5
H = 6h
window = (t, t+H]
```

Despite the historical name `y_event`, this is a future storm-condition target, not an event-onset-only target.

## 4. Temporal Information Rule

For prediction time `t`:

```text
information_cutoff = t - 1h
maximum_feature_information_time <= information_cutoff
```

For raw OMNI timestamp `s`:

```text
period_start = s
period_end   = s + 1h
eligible     = period_end <= information_cutoff
```

Predictor-side Kp uses:

```text
kp_asof(q) = most recent canonical Kp interval with interval_end <= q
```

Future retrospective Kp may define the target and event truth but may never enter the predictor matrix.

## 5. Primary Feature Manifest

The frozen primary feature family order is:

```text
raw
rolling
persistence
dynamics
interactions
```

with:

```text
10 + 60 + 5 + 15 + 3 = 93 features
```

No feature-layer forward fill, backward fill, nearest-row substitution, implicit interpolation, or fallback to an older raw timestamp is permitted under the frozen Phase 0 contract.

## 6. Primary Predictor Sources

Included:

```text
OMNI solar-wind measurements
causal Kp history
```

Excluded from the primary causal experiment:

```text
AE
Dst
CDAW/LASCO CME-derived predictors
```

CDAW acquisition, parsing, tests, and audit code are retained as reproducible research/audit infrastructure.

## 7. Temporal Validation

| Split | Period |
|---|---|
| Initial Train | 1996–2016 |
| Validation 1 | 2017–2018 |
| Train 2 | 1996–2018 |
| Validation 2 | 2019–2020 |
| Train 3 | 1996–2020 |
| Validation 3 | 2021 |
| Final Test | 2022–2025 |

The Final Test is protected and single-use.

## 8. Current Project Status

```text
Phase 0 — causality and temporal infrastructure     COMPLETE
Phase 1 — dataset construction and temporal splits NEXT
Phase 2 — baselines                                PENDING
Phase 3 — feature screening                        PENDING
Phase 4 — imbalance experiments                    PENDING
Phase 5 — model selection                          PENDING
Phase 6 — OOF operational threshold selection      PENDING
Phase 7 — horizon/severity experiments              PENDING
Phase 8 — protected Final Test                      LOCKED
Phase 9 — interpretation/scientific audit           PENDING
```

No model result is final while the required downstream development and protected evaluation stages remain incomplete.

## 9. Canonical Phase 0 Documentation

```text
MASTER_PROTOCOL_v1.3.md
docs/data_contract.md
docs/temporal_cutoff.md
docs/event_definition.md
docs/alert_definition.md
docs/cme_availability.md
docs/feature_contract.md
docs/feature_definition.md
docs/target_definition.md
docs/phase0_temporal_integrity.md
docs/phase0_completion_checklist.md
```

Historical protocol versions remain unchanged.

## 10. Next Step

Phase 1 begins with canonical dataset construction and temporal splitting.

The first Phase 1 artifact should assemble an hourly prediction grid containing:

```text
prediction_time
93 causal predictor features
canonical target
temporal/audit metadata
```

without fitting or selecting a model.
