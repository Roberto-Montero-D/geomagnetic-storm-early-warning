# Geomagnetic Storm Early Warning System

**Status:** Protocol Frozen — Phase 0 Implementation in Progress  
**Protocol:** `MASTER_PROTOCOL_v1.2.md`  
**Primary Horizon:** 6 hours  
**Primary Storm Threshold:** Kp ≥ 5

## 1. Project Overview

This repository contains the implementation of a scientifically controlled early warning system for geomagnetic storms. The objective is **not** to build a generic Kp classifier.

> Given only information that would have been available at prediction time `t`, can the system reliably warn whether geomagnetic storm conditions (`Kp >= T`) will occur within the next `H` hours?

Version 1.2 incorporates source-verification amendments established during Phase 0 before model training: the 1996 historical extension, verified OMNI timestamp semantics, conservative Kp availability, and exclusion of retrospective AE/Dst from the primary causal feature set.

## 2. Scientific Principles

1. Causal temporal information.
2. Explicit event definition.
3. Operational alert episodes.
4. Chronological validation.
5. Protected final test.

## 3. Current Implementation Status

### Completed

- [x] Master protocol and central configuration
- [x] Repository scaffold and Data Contract specification
- [x] Event and alert specifications
- [x] OMNI timestamp/source audit
- [x] OMNI `.fmt` + `.lst` loader
- [x] OMNI schema and timeline validation
- [x] Real 1996–2025 OMNI smoke test
- [x] Kp timestamp and 3-hour interval semantics
- [x] Causal Kp normalization and causality tests
- [x] Real 1996–2025 Kp integration test
- [x] AE/Dst availability audit and primary-feature exclusion policy

### Pending

- [ ] CME information-availability audit
- [ ] Generic temporal cutoff infrastructure
- [ ] Dataset construction
- [ ] Event and alert implementation
- [ ] Causal feature pipeline and target builder
- [ ] Full temporal leakage suite
- [ ] Baselines, feature screening, imbalance screening, model selection
- [ ] Walk-forward validation and OOF threshold selection
- [ ] Alternative H/T experiments
- [ ] Protected final test
- [ ] Error analysis and scientific audit

## 4. Repository Structure

```text
geomagnetic-storm-early-warning/
├── MASTER_PROTOCOL_v1.2.md
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── docs/
│   ├── data_contract.md
│   ├── event_definition.md
│   └── alert_definition.md
├── scripts/
│   └── smoke_test_omni_kp.py
├── src/
│   ├── analysis/
│   ├── data/
│   │   ├── omni.py
│   │   └── kp.py
│   ├── definitions/
│   ├── evaluation/
│   └── models/
└── tests/
    ├── test_omni_loader.py
    └── test_kp_causality.py
```

Raw `data/` is local and intentionally excluded from version control by `.gitignore`.

## 5. Primary Experimental Configuration

| Parameter | Primary value |
|---|---:|
| Storm threshold T | 5 |
| Forecast horizon H | 6 h |
| Event separation Z | 6 h |
| Alert cooldown C | 3 h |
| Maximum FAR/day | 0.2 |
| Protocol information cutoff | `t - 1h` |

For OMNI:

```text
period_end = raw_timestamp + 1h
period_end <= t - 1h
```

## 6. Temporal Information Rule

Phase 0.1 verified that raw OMNIWeb timestamps mark the start of represented hourly intervals `[s, s+1h)`. For prediction at 14:00, the 12:00 row ending at 13:00 is allowed; the 13:00 row ending at 14:00 is not.

## 7. Kp Causal Availability

Kp is a 3-hour index repeated across three hourly OMNI rows. The implementation collapses this representation into canonical intervals. `kp_asof(q)` returns the most recent interval satisfying `interval_end <= q`.

Primary lags are `kp_asof(t-1h)`, `kp_asof(t-3h)`, `kp_asof(t-6h)`, `kp_asof(t-12h)`, and `kp_asof(t-24h)`.

This is a conservative historical availability approximation; the historical GFZ nowcast stream is not reconstructed.

## 8. Target and Event Ground Truth

Future retrospective Kp is used only to construct the target. Predictor-side `kp_asof()` does not apply to target/event truth.

The primary target is:

```text
y_event(t) = max(Kp[t+1:t+H]) >= T
```

Storm onset is the start of the first canonical 3-hour Kp interval satisfying `Kp >= T`. Event segmentation uses an hourly-expanded ground-truth state so `Z=6h` remains the frozen separation rule.

## 9. Alert Definition

An alert occurs when `P(event within H hours) >= tau`. Alert-producing timestamps are grouped with `C=3h`. Operational evaluation is performed at alert-episode and storm-event level.

## 10. Validation Strategy

| Split | Period |
|---|---|
| Initial Train | 1996–2016 |
| Validation 1 | 2017–2018 |
| Train 2 | 1996–2018 |
| Validation 2 | 2019–2020 |
| Train 3 | 1996–2020 |
| Validation 3 | 2021 |
| Final Test | 2022–2025 |

The historical training start was extended from 2008 to 1996 during Phase 0 after source coverage and data quality were verified. Validation periods and the protected final-test period were not changed.

## 11. Metrics

Primary operational metrics are Event Recall, FAR/day, and Lead Time. Late Detection Rate, Precision, PR-AUC, and reliability/calibration are secondary or diagnostic as specified by the protocol.

## 12. Data Sources

The current OMNI subset contains magnetic-field quantities, solar-wind temperature, density, speed, flow pressure, electric field, plasma beta, Alfvén Mach number, Kp, Dst, and AE.

AE and Dst are preserved during raw ingestion but excluded from the primary causal feature set. CME information is admitted only when historical real-time availability can be demonstrated; that audit remains Phase 0.3.

## 13. Reproducibility

Phase 0 validation currently includes:

- 262,992 continuous hourly OMNI rows from 1996–2025;
- zero missing or duplicate OMNI timestamps;
- 87,664 canonical Kp intervals;
- zero missing canonical Kp intervals;
- automated OMNI loader tests;
- automated Kp causality tests;
- full real-data OMNI → Kp integration smoke test.

## 14. Development Order

```text
Frozen Protocol
      ↓
Data Contract
      ↓
OMNI / Geomagnetic-Index Audit
      ↓
CME Availability Audit
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

## 15. Status

This repository is currently in **Phase 0 implementation and causal-data validation**. OMNI ingestion and Kp causal normalization are implemented and validated. CME availability, event/alert implementation, the complete causal feature pipeline, and downstream model-development phases remain pending.
