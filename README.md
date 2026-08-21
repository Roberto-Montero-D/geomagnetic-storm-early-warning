# Geomagnetic Storm Early Warning System

**Status:** Protocol Frozen — Phase 0 Implementation in Progress  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Primary Horizon:** 6 hours  
**Primary Storm Threshold:** Kp ≥ 5

## 1. Project Overview

This repository implements a scientifically controlled early-warning system for geomagnetic storms.

The objective is **not** to build a generic Kp classifier. The operational question is:

> Given only information that would have been available at prediction time `t`, can the system issue a reliable warning that geomagnetic storm conditions will occur within the next `H` hours?

Version 1.3 incorporates the completed Phase 0 source-availability audits performed before model training. These amendments establish the 1996 historical coverage, verified OMNI timestamp semantics, conservative Kp availability, exclusion of retrospective AE/Dst from the primary causal feature set, and exclusion of CDAW/LASCO CME-derived predictors because uniform historical candidate-event availability could not be demonstrated.

The primary predictor universe is therefore frozen to **causally eligible OMNI solar-wind measurements plus conservative causal Kp history**.

## 2. Scientific Principles

1. **Causal temporal information** — features may only use information available by the prediction cutoff.
2. **Explicit event definition** — storm events are defined independently of the model.
3. **Operational alert definition** — hourly alerts are grouped into alert episodes and evaluated at event level.
4. **Temporal validation** — training, validation, OOF threshold selection, and testing are chronological.
5. **Final-test protection** — 2022–2025 remains untouched during development.

## 3. Current Implementation Status

### Completed

- [x] Master protocol and Phase 0 amendment history
- [x] Central configuration
- [x] Data Contract specification
- [x] Event definition specification
- [x] Alert definition specification
- [x] OMNI timestamp/source audit
- [x] OMNI loader
- [x] Kp canonical interval and causal mapping implementation
- [x] Kp causality tests
- [x] AE/Dst historical-availability audit
- [x] CDAW/LASCO CME historical-availability audit
- [x] Full 1996–2025 CDAW height-time audit
- [x] CME measurement-vs-candidate causality assessment
- [x] CME primary-feature exclusion policy
- [x] Primary source universe frozen to OMNI + causal Kp

### Pending

- [ ] Generic temporal cutoff infrastructure
- [ ] Event detection implementation and tests
- [ ] Alert episode implementation and tests
- [ ] Causal rolling/persistence/dynamic/interaction feature pipeline
- [ ] Target construction
- [ ] Global temporal leakage/future-mutation tests
- [ ] Dataset construction
- [ ] Baseline models
- [ ] Feature screening
- [ ] Imbalance experiments
- [ ] Model selection
- [ ] Walk-forward validation
- [ ] OOF threshold selection
- [ ] Alternative horizon/severity experiments
- [ ] Protected final test
- [ ] Error analysis and scientific audit

No model result is final while the required implementation and protected evaluation stages remain incomplete.

## 4. Primary Experimental Configuration

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

## 5. Primary Predictor Sources

The primary operational model is restricted to:

```text
causally eligible OMNI solar-wind measurements
+
conservative causal Kp history
```

Excluded from the primary causal feature matrix:

- **AE and Dst:** retrospective historical values cannot be demonstrated to consistently equal values available at historical prediction time.
- **CDAW/LASCO CME:** timestamped height-time measurements are technically suitable for causal reconstruction, but the retrospectively curated candidate-event universe does not provide uniform historical availability semantics across 1996–2025.

CME acquisition, parsing, tests, and audit results are retained for reproducibility and possible future research extensions.

See `docs/cme_availability.md`.

## 6. Temporal Information Rule

For prediction time `t`:

```text
information_cutoff = t - 1h
```

For raw OMNI timestamp `s`:

```text
period_start = s
period_end   = s + 1h
eligible     = period_end <= information_cutoff
```

This verified interval rule must be used by all primary OMNI-derived features.

## 7. Kp Causality

Kp is treated as a canonical 3-hour index.

Predictor-side Kp uses:

```text
kp_asof(q) = most recent canonical Kp interval with interval_end <= q
```

Primary lag family:

```text
1h, 3h, 6h, 12h, 24h
```

Retrospective Kp remains valid for event and target ground truth.

## 8. CME Phase 0.3 Result

The full CDAW height-time audit produced:

```text
successful .yht records:                 42,422
>=3 measurement trajectories:            41,705
fraction >=3:                            98.3098%
third point within 6h:                   99.8945%
explicit retrospective insertions:        3,410
duplicate-timestamp trajectories:             21
non-monotonic trajectories:                  17
invalid heights:                              0
```

These results demonstrate that measurement-level causal reconstruction is technically feasible.

CME was nevertheless excluded because **candidate-event historical availability** could not be established uniformly enough for the primary causal standard. The decision was made before model training and was not based on predictive performance.

## 9. Event and Alert Definitions

Storm events are defined from retrospective canonical Kp according to `docs/event_definition.md`.

Operational alert episodes and their association with events are defined in `docs/alert_definition.md`.

Primary operational evaluation uses:

```text
Event Recall
FAR/day
Lead Time
```

with:

```text
FAR/day <= 0.2
```

## 10. Temporal Validation

| Split | Period |
|---|---|
| Initial Train | 1996–2016 |
| Validation 1 | 2017–2018 |
| Train 2 | 1996–2018 |
| Validation 2 | 2019–2020 |
| Train 3 | 1996–2020 |
| Validation 3 | 2021 |
| Final Test | 2022–2025 |

The Final Test is single-use and must not influence source selection, feature selection, model selection, hyperparameters, balancing, or threshold selection.

## 11. Repository Structure

The Phase 0.3 target repository structure includes:

```text
geomagnetic-storm-early-warning/
├── MASTER_PROTOCOL_v1.3.md
├── README.md
├── config/
│   └── config.yaml
├── docs/
│   ├── data_contract.md
│   ├── event_definition.md
│   ├── alert_definition.md
│   └── cme_availability.md
├── scripts/
│   ├── smoke_test_omni_kp.py
│   ├── audit_cdaw_yht.py
│   └── retry_cdaw_failures.py
├── src/
│   └── data/
│       ├── omni.py
│       ├── kp.py
│       └── cme_cdaw.py
└── tests/
    ├── test_omni_loader.py
    ├── test_kp_causality.py
    └── test_cme_cdaw.py
```

The CDAW implementation/audit files are **audit/research infrastructure** and must not be imported by the primary feature pipeline.

## 12. Development Order

```text
Frozen Protocol
      ↓
Data Contract
      ↓
OMNI / Geomagnetic-Index Audit
      ↓
CME Availability Audit
      ↓
Primary Source Universe Frozen
      ↓
Event / Alert Implementation
      ↓
Causal Feature Infrastructure
      ↓
Leakage Tests
      ↓
Dataset Construction
      ↓
Models / Walk-Forward / OOF
      ↓
Protected Final Test
      ↓
Scientific Audit
```

## 13. Reproducibility

The project preserves source-semantic audits, protocol versions, configuration, canonical definitions, source loaders, causality tests, and audit scripts.

Historical protocol versions remain unchanged.

CDAW/LASCO code is retained as reproducible evidence supporting the Phase 0.3 exclusion decision.

## 14. Current Status

The repository is currently in **Phase 0 implementation and causal-data validation**.

OMNI ingestion, Kp causal normalization, AE/Dst availability auditing, and the CME availability/causality audit are complete.

The primary predictor universe is frozen to **OMNI solar-wind measurements plus causal Kp history**.

Event/alert implementation, the complete causal feature pipeline, global leakage validation, and downstream model-development phases remain pending.
