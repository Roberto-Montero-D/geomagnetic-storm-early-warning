# MASTER PROTOCOL
## Geomagnetic Storm Early Warning System

**Version:** 1.2 (Frozen — Phase 0 Amendment)  
**Date:** August 20, 2026  
**Status:** Frozen. Phase 0 amendments were made before model training or performance inspection.

---

## v1.2 Amendment Note

Version 1.2 incorporates Phase 0 source-verification findings established before model training and before any performance-driven model selection.

The amendments are:

1. Historical training coverage is extended from 2008 to 1996.
2. The raw OMNI timestamp is corrected to the start of the represented hourly interval.
3. Kp is treated as a 3-hour index with conservative causal availability.
4. Retrospective AE and Dst are excluded from the primary causal feature set.
5. Validation periods and the protected 2022–2025 final test are unchanged.

These amendments correct source-availability and experimental-coverage assumptions and were not motivated by model results.

---

## 1. PHILOSOPHY AND OBJECTIVE

### 1.1 Core Philosophy

> This project builds an early warning system for geomagnetic storms. It is **not** a Kp classifier. It is a system that must be causally correct, operationally useful, scientifically interpretable, and temporally robust.

### 1.2 Question the System Answers

> "Will there be geomagnetic storm conditions (Kp >= T) within the next H hours?"

### 1.3 Definition of Success

> "The system detects X% of events with a median lead time of Y hours, producing Z false alarms per day."

### 1.4 Guiding Principles

1. **Causality:** Only information available in real time is used.
2. **No leakage:** All features are audited with a Data Contract.
3. **No data snooping:** Decisions are made before seeing results.
4. **Temporal robustness:** Walk-forward validation, not a single partition.
5. **Interpretability:** SHAP and case studies explain predictions.

### 1.5 Freeze Rule

> Once implementation begins, T, H, Z, C, event definition, primary metric, splits, etc. are **not** modified based on results. Only implementation bugs, verified source-semantic corrections, or inconsistencies with the Data Contract may be corrected, provided the correction is documented and is not motivated by model performance.

---

## 2. FUNDAMENTAL DEFINITIONS (PHASE 0)

### 2.1 Timestamp and Availability Convention

**Fundamental rule:**

> For prediction time `t`, feature information must satisfy `maximum_feature_information_time <= t - 1h`.

Phase 0.1 verified that the raw OMNIWeb timestamp marks the **start** of the represented hourly interval.

For raw timestamp `s`:

```text
period_start = s
period_end   = s + 1h
period       = [s, s + 1h)
```

An OMNI record is eligible for prediction time `t` only when:

```text
period_end <= t - 1h
```

Example:

```text
prediction_time = 14:00
information_cutoff = 13:00

12:00 OMNI row -> [12:00, 13:00) -> allowed
13:00 OMNI row -> [13:00, 14:00) -> not allowed
```

### 2.2 Data Contract

| Feature | Source | Verified Temporal Meaning | Primary Feature Policy |
|---|---|---|---|
| Bz | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Bt | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| V | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Density | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Pressure | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| AE | OMNI | Retrospective hourly index | Excluded from primary causal features |
| Dst | OMNI | Retrospective hourly index | Excluded from primary causal features |
| Kp | OMNI / GFZ | 3-hour interval repeated over hourly rows | Use conservative `kp_asof()` mapping |
| CME | Catalog | Source-dependent | Only if historical availability is demonstrated |

### 2.3 Kp Causal Availability

Kp is a 3-hour geomagnetic index. OMNI repeats each 3-hour Kp value across the three hourly rows belonging to that interval. Raw OMNI Kp uses the `Kp × 10` integer encoding and is converted to standard Kp units during normalization.

For predictor-side Kp, `kp_asof(q)` returns the Kp value from the most recent canonical 3-hour interval satisfying:

```text
interval_end <= q
```

The primary Kp lag features are:

```text
Kp_lag_1h(t)  = kp_asof(t - 1h)
Kp_lag_3h(t)  = kp_asof(t - 3h)
Kp_lag_6h(t)  = kp_asof(t - 6h)
Kp_lag_12h(t) = kp_asof(t - 12h)
Kp_lag_24h(t) = kp_asof(t - 24h)
```

This is a conservative historical approximation and does not attempt to reconstruct the historical GFZ nowcast stream before interval completion.

Kp used for target and event truth remains retrospective historical ground truth and is not passed through predictor-side `kp_asof()`.

### 2.4 AE and Dst Availability

AE and Dst are retained in raw OMNI ingestion but are excluded from the primary causal feature set.

The historical OMNI series contains retrospective provisional, quick-look, and/or final products whose values cannot be shown to consistently match the values available at historical prediction time.

```text
raw ingestion:                 allowed
exploratory analysis:          allowed
primary causal feature set:    excluded
```

This decision was made during Phase 0 before model training and was not based on predictive performance.

### 2.5 Specific Rule for CME (CRITICAL)

The rule remains:

> `cme_information_available_at_t == True`

Usable CME characteristics are permitted only when the corresponding information was available at `t`. Phase 0.3 must verify historical availability semantics before CME features are admitted to the operational feature set.

### 2.6 Target Definition

**Primary target:**

> `y_event(t) = max(Kp[t+1 : t+H]) >= T`

| Parameter | Value | Description |
|---|---:|---|
| T | 5 | Kp threshold for storm conditions |
| H | 6 | Prediction horizon in hours |

Phase 7 experiments remain pre-specified at `T = 5, 6, 7` and `H = 3, 6, 12, 24` as defined below.

All models and baselines use exactly the same target for a given H/T experiment.

### 2.7 Event Definition (Storm Episodes)

For event and target ground truth, Kp is interpreted from its canonical 3-hour interval representation and expanded onto the project hourly evaluation timeline when the `Z` rule is evaluated.

| Rule | Value |
|---|---|
| Event start | Start of first Kp interval where `Kp >= T` |
| Event end | Last active storm hour before `Kp < T` for `Z` consecutive hourly ground-truth states |
| Z | 6 hours |
| Separate events | Independent after at least Z valid consecutive below-threshold hours |

With complete aligned Kp data, `Z = 6h` corresponds to two complete consecutive 3-hour Kp intervals below threshold.

Each real event is counted once regardless of the number of alert episodes that overlap it.

### 2.8 Alert and Alert Episode Definition

An hourly alert occurs when `P(storm within H hours) >= tau`. Alert-producing timestamps separated by `gap <= C` belong to the same episode; `gap > C` starts a new episode.

For the primary system, `C = 3h`.

Alert classification uses the episode's `first_alert_time`:

| Condition | Classification |
|---|---|
| `storm_start - H <= first_alert < storm_start` | Early Detection |
| `storm_start <= first_alert <= storm_end` | Late Detection |
| No valid event association | False Alarm |

If multiple Early Detection episodes qualify for one storm, the earliest qualifying episode is used for storm-level detection and lead time. If no Early Detection exists, the earliest qualifying Late Detection is used. Additional qualifying episodes do not increase Event Recall.

### 2.9 Official Metrics

| Metric | Definition |
|---|---|
| Event Recall | Detected events / evaluable real events |
| FAR/day | False alert episodes / evaluation duration in days |
| Lead Time | `storm_start - first_alert` for Early Detections |
| Late Detection Rate | Late Detections / detected events |
| Precision | Correct episodes / total episodes |
| PR-AUC | Area under Precision-Recall curve |
| Reliability | Calibration curve and Brier Score |

Primary operational constraint:

```text
FAR/day <= 0.2
```

The exact FAR/day denominator treatment in the presence of invalid prediction timestamps must be frozen during Phase 0 before threshold optimization.

---

## 3. TEMPORAL SPLITS (PHASE 1)

### 3.1 Split Structure

| Split | Period | Purpose |
|---|---|---|
| Initial Train | 1996–2016 | Screening |
| Validation 1 | 2017–2018 | Screening |
| Train 2 | 1996–2018 | Walk-forward fold 1 |
| Validation 2 | 2019–2020 | Walk-forward fold 1 |
| Train 3 | 1996–2020 | Walk-forward fold 2 |
| Validation 3 | 2021 | Walk-forward fold 2 |
| Final Test | 2022–2025 | ONE TIME, AT THE END |

The historical start was extended from 2008 to 1996 during Phase 0 after verifying continuous OMNI coverage and early-period data quality. This occurred before model training or performance inspection. Validation periods and the protected final-test period were not changed.

### 3.2 Split Rules

1. Never mix future data into training.
2. The Final Test is used once and nothing is modified afterward.
3. Validation and final-test periods remain fixed.

### 3.3 Solar-Cycle Context and Screening Limitation

- **Initial Train (1996–2016):** includes most of Solar Cycle 23 and the majority of Solar Cycle 24 up to its declining phase, providing broader solar-cycle regime coverage than the original 2008 start.
- **Validation 1 (2017–2018):** late Solar Cycle 24 / transition toward Solar Cycle 25.
- **Validation 2 (2019–2020):** Solar Cycle 24 minimum / beginning of Solar Cycle 25.
- **Validation 3 (2021):** rising Solar Cycle 25.
- **Final Test (2022–2025):** strong rise and maximum of Solar Cycle 25.

Initial screening on Validation 1 may remain sensitive to that physical regime. Walk-forward confirmation is the safeguard against selecting a configuration that performs well only in the screening period.

---

## 4. FEATURES (PHASE 1)

### 4.1 Feature Layers

| Layer | Features | Purpose |
|---|---|---|
| Raw | Bz, Bt, V, Density, Pressure, Kp_lag_1h/3h/6h/12h/24h | Physical and geomagnetic history |
| Rolling | mean/min/std over 3/6/12/24h as specified | Averages, minima, variability |
| Persistence | Bz and V threshold durations | Duration of adverse conditions |
| Dynamics | 1h/3h deltas and 3h slopes | Changes and trends |
| CME | Causally verified CME temporal features only | Recent CME activity |
| Interactions | Bz_neg×V, Bz_neg×Density, Pressure×V | Variable combinations |

AE and Dst remain in raw ingestion for audit and exploratory analysis but are excluded from the primary causal feature set.

### 4.2 Causal Implementation Rule

```text
period_start = raw_timestamp
period_end = raw_timestamp + 1h
information_cutoff = prediction_time - 1h
eligible = period_end <= information_cutoff
```

Rolling, persistence, dynamic, and interaction features must be constructed only from observations that already satisfy this eligibility rule. No unconditional `.shift(1)` operation is part of the protocol.

---

## 5. BASELINES (PHASE 2)

All baselines produce `y_hat_event(t)` for comparison with the same target.

| Baseline | Model | Purpose |
|---|---|---|
| B0: Persistence | `1 if kp_asof(t-1h) >= T else 0` | Simple persistence |
| B1: Physical | Bz < -X AND V > Y | Simple physical rule |
| B2: Logistic | Logistic Regression (raw primary features) | Linear model |
| B3: ExtraTrees | ExtraTrees (raw primary features, no balancing) | Simple non-linear model |

Evaluation: Event Recall, FAR/day, Lead Time.

---

## 6. FEATURE ENGINEERING (PHASE 3)

### 6.1 Screening (Validation 1)

| Experiment | Features |
|---|---|
| A | Raw physical + causal Kp lag family |
| B | A + Rolling |
| C | B + Persistence |
| D | C + Dynamics |
| E | D + causally verified CME temporal features |
| F | E + Interactions |

Procedure: train ExtraTrees without balancing on Initial Train, evaluate on Validation 1, and record Event Recall, FAR/day, and PR-AUC.

### 6.2 Confirmation (Walk-Forward)

The 2–3 best feature sets proceed to walk-forward. Selection is based on consistency across folds and Event Recall subject to `FAR/day <= 0.2`.

---

## 7. IMBALANCE HANDLING (PHASE 4)

| Experiment | Strategy | Parameters |
|---|---|---|
| 1 | No balancing | Threshold optimized |
| 2 | Class weighting | pos_weight = 1, 3, 5, 10, 20, 50 |
| 3 | Random undersampling | 10:1, 5:1, 2:1 |
| 4 | SMOTE | k=3, 5, 7 |
| 5 | Borderline-SMOTE | k=3, 5, 7 |
| 6 | SMOTE-ENN | — |

Top strategies proceed to walk-forward confirmation. Calibration and threshold stability are diagnostic analyses.

---

## 8. MODEL SELECTION (PHASE 5)

| Model | Hyperparameters |
|---|---|
| ExtraTrees | n_estimators = 100, 200, 500; max_depth = 10, 20, None |
| LightGBM | learning_rate = 0.01, 0.05, 0.1; num_leaves = 31, 63, 127 |
| XGBoost | learning_rate = 0.01, 0.05, 0.1; max_depth = 3, 6, 9 |

Stacking is optional only if ExtraTrees and LightGBM show complementary errors.

---

## 9. OPERATIONAL OPTIMIZATION (PHASE 6)

Generate temporally ordered OOF predictions for each validation fold and retain timestamp, probability, target, storm_id, and fold.

For thresholds 0.01–0.99 in steps of 0.01, construct canonical alert episodes and calculate Event Recall and FAR/day. Select the **lowest threshold** satisfying:

```text
FAR/day <= 0.2
```

If no candidate threshold satisfies the operational constraint, the model fails the constraint; the constraint is not relaxed retrospectively.

Report the global threshold, fold-specific thresholds, and the diagnostic threshold range corresponding to `0.15 <= FAR/day <= 0.20`. The global OOF-selected threshold remains the operational threshold.

---

## 10. HORIZON AND SEVERITY EXPERIMENTS (PHASE 7)

Feature set, model, balancing strategy, hyperparameters, and alert rules remain frozen. Only H, T, and the OOF-recalibrated threshold change.

Alternative H/T experiments evaluate the generalization of the frozen primary system and are not independently re-optimized model-development tracks.

Pre-specified combinations include H = 3, 6, 12, 24 for T = 5 and H = 6 for T = 6 and 7.

---

## 11. FINAL TEST VALIDATION (PHASE 8)

Before touching Test, feature set, model/hyperparameters, balancing strategy, threshold, C, Z, H, and T are frozen.

1. Retrain with all permitted development data through 2021.
2. Do not inspect Test during training or development.
3. Run once on Test (2022–2025).
4. Calculate all protocol metrics and confidence intervals.

Event Recall uses event-level bootstrap, Lead Time uses bootstrap on Early Detections, and FAR/day uses temporal block bootstrap.

---

## 12. ERROR ANALYSIS AND INTERPRETATION (PHASE 9)

Analyze false negatives, false alarms, SHAP/feature importance, and 3–5 representative storm case studies. False alarms may be categorized for scientific interpretation as physically plausible/CME-associated versus model/noise false alarms, but this categorization never changes official FAR/day.

---

## 13. SCIENTIFIC ABLATION EXPERIMENT (PHASE 10)

| Experiment | Features |
|---|---|
| M0 | Only causal Kp persistence (`kp_asof(t-1h)`) |
| M1 | Causal Kp lag family |
| M2 | Solar wind + causally verified CME temporal features |
| M3 | M1 + M2 |

M0–M3 are a pre-specified confirmatory ablation experiment. No feature, hyperparameter, threshold, or model modification is performed after observing final-test results.

---

## 14. PHASE 0.1 / 0.2 VERIFICATION RECORD

Verified on the project OMNI subset:

```text
Coverage:             1996-01-01 00:00 through 2025-12-31 23:00
Hourly rows:          262,992
Missing timestamps:   0
Duplicate timestamps: 0
Canonical Kp bins:    87,664
Missing Kp bins:      0
```

The OMNI loader and Kp causal normalization are covered by automated unit tests and a full-dataset integration smoke test.

---

## 15. KEY DECISIONS — FINAL SUMMARY

| Decision | Value |
|---|---|
| Primary target | Event-in-Window, T=5, H=6h |
| Z | 6h |
| C | 3h |
| Max FAR/day | 0.2 |
| Information cutoff | `t - 1h` |
| OMNI eligibility | `period_end <= t - 1h` |
| Kp predictor availability | completed canonical interval via `kp_asof()` |
| AE/Dst | excluded from primary causal feature set |
| Final threshold | global OOF predictions |
| Event counting | each event counted once |
| Final Test | 2022–2025, protected, single use |

---

## 16. CHANGE LOG

### v1.2 — August 20, 2026

Phase 0 amendments made before model training or performance inspection:

- extended historical training start from 2008 to 1996;
- corrected raw OMNI timestamp semantics to start-of-period;
- introduced explicit `period_end` causal eligibility;
- formalized conservative 3-hour Kp predictor availability;
- excluded retrospective AE and Dst from the primary causal feature set;
- preserved all validation periods and the protected 2022–2025 final test.

### v1.1 — August 17, 2026

Original frozen protocol prior to Phase 0 source verification.

---

## 17. FREEZE DECLARATION

> **This document constitutes the master protocol of the project. All methodological decisions contained herein are final and will not be modified based on results observed during execution. Any deviation requires documented justification based exclusively on implementation issues, verified source semantics, or data inconsistencies, not on model performance.**

---

**End of Master Protocol**
