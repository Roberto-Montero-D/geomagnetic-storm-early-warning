# Geomagnetic Storm Early Warning System

**Status:** Protocol Frozen — Implementation in Progress  
**Protocol:** `MASTER_PROTOCOL_v1.1.md`  
**Primary Horizon:** 6 hours  
**Primary Storm Threshold:** Kp ≥ 5

## 1. Project Overview

This repository contains the implementation of a scientifically controlled early warning system for geomagnetic storms.

The objective is **not** to build a generic Kp classifier.

The system is designed to answer the operational question:

> Given only information that would have been available at prediction time `t`, can the system reliably warn wether geomagnetic storm conditions(`Kp >= T`) will occur within the next `H` hours?

The project therefore treats geomagnetic storm prediction as a **temporal forecasting and operational alert problem**, rather than as ordinary binary classification.

The complete methodology is defined in:

```text
MASTER_PROTOCOL_v1.1.md
```

The protocol is frozen before model development and evaluation. Implementation decisions must remain consistent with that document.

---

## 2. Scientific Principles

The implementation follows five central principles:

1. **Causal temporal information**
   - Features may only use information available by the prediction time.
   - Future observations may only be used to construct the target.

2. **Explicit event definition**
   - A storm event is defined independently of the model.
   - Individual hourly positive labels are not treated as independent storm events.

3. **Operational alert definition**
   - Consecutive alerts are grouped into alert episodes.
   - Detection performance is evaluated at the event level.

4. **Temporal validation**
   - Training, validation, out-of-fold threshold selection, and final testing follow chronological splits.
   - Random train/test splitting is not used for the primary experiment.

5. **Final test protection**
   - The final test period remains untouched during model development, feature selection, threshold selection, and model comparison.

---

## 3. Current Implementation Status

The repository is intentionally being developed incrementally.

### Completed

- [x] Master protocol
- [x] Central configuration
- [x] Repository scaffold
- [x] Data Contract specification
- [x] Event definition specification
- [x] Alert definition specification

### Pending

- [ ] Verify OMNI timestamp convention
- [ ] Verify Kp timestamp convention
- [ ] Define and verify CME information availability
- [ ] Implement data ingestion
- [ ] Implement dataset construction
- [ ] Implement event detection
- [ ] Implement alert episode construction
- [ ] Implement causal feature pipeline
- [ ] Implement target construction
- [ ] Implement temporal leakage tests
- [ ] Implement baseline models
- [ ] Implement feature screening
- [ ] Implement model selection
- [ ] Implement walk-forward validation
- [ ] Implement OOF threshold selection
- [ ] Implement alternative horizon/threshold experiments
- [ ] Run the protected final test
- [ ] Run error analysis and scientific audit

**No model result should be considered final while the items above remain incomplete.**

---

## 4. Repository Structure

The current repository contains the methodological and implementation scaffold.

```text
geomagnetic-storm-early-warning/
│
├── MASTER_PROTOCOL_v1.1.md
├── README.md
├── requirements.txt
│
├── config/
│   └── config.yaml
│
├── docs/
│   ├── data_contract.md
│   ├── event_definition.md
│   └── alert_definition.md
│
├── src/
│   ├── analysis/
│   ├── definitions/
│   ├── evaluation/
│   └── models/
│
└── tests/
```

Additional directories such as `data/`, `scripts/`, `notebooks/`, and `results/` will be introduced only when their implementation is required.

The documentation must describe the repository as it actually exists. Planned components must not be presented as currently implemented.

---

## 5. Primary Experimental Configuration

The primary experiment is defined by the frozen protocol:

| Parameter | Primary value |
|---|---:|
| Storm threshold `T` | 5 |
| Forecast horizon `H` | 6 h |
| Event separation parameter `Z` | 6 h |
| Alert persistence `C` | 3 h |
| Maximum FAR/day | 0.2 |
| Latest permissible observation | `t - 1 h` |

These values are controlled centrally through:

```text
config/config.yaml
```

They must not be silently overridden by individual scripts or notebooks.

---

## 6. Temporal Information Rule

For a prediction made at time `t`, the feature set must only contain information that would have been available by:

```text
t - 1 hour
```

The exact interpretation of source timestamps must be verified before the feature pipeline is implemented.

In particular, the implementation must first establish whether the timestamp associated with an OMNI record represents the beginning, end, or another convention for the represented observation interval.

This is a Phase 0 validation task.

Until this has been verified, no feature transformation should assume a timestamp interpretation that has not been documented and tested.

---

## 7. Target Definition

For the primary experiment, the target represents whether geomagnetic storm conditions (Kp >= T) will occur within the forecast horizon.

The target is therefore constructed from future Kp information and is kept logically separate from feature construction.

Conceptually:

```text
Past / currently available information
                ↓
             Features
                ↓
          Prediction time t
                ↓
       Future interval t+1 … t+H
                ↓
              Target
```

Future Kp observations must never enter the feature matrix.

---

## 8. Event Definition

A geomagnetic storm event begins at the first hour at which:

```text
Kp ≥ T
```

with:

```text
T = 5
```

An event ends according to the consecutive-below-threshold rule defined in:

```text
docs/event_definition.md
```

The event definition is independent of the model and must be implemented deterministically.

---

## 9. Alert Definition

An alert is issued when:

```text
P(event within H hours) ≥ τ
```

where `τ` is selected using the protocol-defined out-of-fold procedure.

Consecutive alert hours are grouped into alert episodes using:

```text
C = 3 hours
```

Operational evaluation is performed at the alert-episode and storm-event level rather than by treating every hourly prediction as an independent warning.

The complete definition is documented in:

```text
docs/alert_definition.md
```

---

## 10. Validation Strategy

The primary experiment uses chronological validation.

The protocol includes:

- historical training periods,
- validation periods,
- out-of-fold predictions for threshold selection,
- a protected final test period.

The final test period must not influence:

- feature selection,
- model selection,
- hyperparameter tuning,
- threshold selection,
- error-driven methodological changes.

This separation is essential for obtaining an unbiased estimate of operational performance.

---

## 11. Metrics

The system is evaluated using operational metrics defined by the master protocol.

These include:

- Event Recall
- False Alarm Rate per day
- Precision
- Lead Time
- Late Detection Rate
- Additional diagnostic metrics where specified by the protocol

Standard classification metrics may be reported as supplementary information, but they do not replace event-level operational evaluation.

---

## 12. Data Sources

The project uses space-weather observations and event information defined by the Data Contract.

The primary variables include:

- IMF magnetic-field quantities
- Solar-wind speed
- Solar-wind density
- Solar-wind pressure
- Geomagnetic indices
- Kp
- CME information where real-time availability can be established

The exact source fields, timestamp semantics, availability rules, and transformations are documented in:

```text
docs/data_contract.md
```

---

## 13. CME Availability

CME information requires special treatment.

A CME must not be considered available merely because the physical event occurred before prediction time.

The implementation must establish whether the relevant CME information would actually have been available to an operational forecaster at time `t`.

This distinction is critical because retrospective catalogs can contain information that was not known in real time.

The reconstruction of this availability is a Phase 0 task.

---

## 14. Reproducibility

All methodological parameters should be controlled through versioned configuration and documented code.

The repository should make it possible to determine:

- which protocol version was used,
- which configuration was used,
- which data sources were used,
- which temporal split was used,
- which feature specification was used,
- which model configuration was used,
- how the threshold was obtained,
- and which test period was evaluated.

No result should depend on undocumented manual intervention.

---

## 15. Development Philosophy

Implementation proceeds in the following order:

```text
Frozen Protocol
      ↓
Data Contract
      ↓
Timestamp Audit
      ↓
Event / Alert Definitions
      ↓
Causal Feature Infrastructure
      ↓
Leakage Tests
      ↓
Dataset Construction
      ↓
Baseline Models
      ↓
Feature Screening
      ↓
Model Selection
      ↓
Walk-Forward Validation
      ↓
OOF Threshold Selection
      ↓
Alternative H/T Experiments
      ↓
Protected Final Test
      ↓
Scientific Audit
```

The order is intentional.

Model performance must not determine the definition of the target, event, alert, temporal split, or information-availability rules.

---

## 16. Status

This repository is currently a **methodological and implementation scaffold**.

The absence of trained models or final results is intentional.

The project should not be considered experimentally complete until the complete pipeline has passed its temporal-causality and leakage tests and the final test has been executed according to `MASTER_PROTOCOL_v1.1.md`.