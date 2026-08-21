# MASTER PROTOCOL
## Geomagnetic Storm Early Warning System

**Version:** 1.3 (Frozen — Phase 0.3 Amendment)  
**Date:** August 21, 2026  
**Status:** Frozen. Phase 0 source-availability amendments were completed before model training or performance inspection.

---

## v1.3 Amendment Note

Version 1.3 incorporates the completed Phase 0.3 CME availability and causality audit.

The Phase 0 amendments now establish:

1. Historical model-development coverage begins in 1996.
2. Raw OMNI timestamps represent the start of hourly averaging intervals.
3. Kp is represented causally through completed canonical 3-hour intervals.
4. AE and Dst are excluded from the primary causal feature set because historical predictor-time value availability cannot be demonstrated consistently.
5. CDAW/LASCO CME-derived predictors are excluded from the primary causal feature set because uniform historical candidate-event availability cannot be demonstrated across 1996–2025.
6. The primary predictor universe is frozen to causally eligible OMNI solar-wind measurements plus conservative causal Kp history.
7. Validation periods and the protected 2022–2025 Final Test remain unchanged.

These amendments were determined from source semantics, availability audits, and leakage prevention before model training or performance-driven feature selection.

The detailed CME investigation is documented in `docs/cme_availability.md`.

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

For prediction time `t`:

```text
maximum_feature_information_time <= t - 1h
```

Phase 0.1 verified that raw OMNIWeb timestamps mark the **start** of the represented hourly interval.

For raw timestamp `s`:

```text
period_start = s
period_end   = s + 1h
period       = [s, s + 1h)
```

An OMNI record is eligible only when:

```text
period_end <= t - 1h
```

### 2.2 Data Contract

| Feature / source | Source | Verified Temporal Meaning | Primary Feature Policy |
|---|---|---|---|
| Bz | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Bt | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| V | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Density | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| Pressure | OMNI | `[s, s+1h)` | Eligible if `period_end <= t-1h` |
| AE | OMNI | Retrospective hourly index | Excluded |
| Dst | OMNI | Retrospective hourly index | Excluded |
| Kp | OMNI / GFZ semantics | Canonical 3-hour intervals | Conservative `kp_asof()` |
| CME | SOHO/LASCO CDAW | Timestamped observations in a retrospectively curated candidate universe | Excluded |

### 2.3 Kp Causal Availability

Kp is a 3-hour geomagnetic index. OMNI repeats each 3-hour Kp value across the three hourly rows belonging to that interval. Raw OMNI Kp uses the `Kp × 10` integer encoding and is converted to standard Kp units during normalization.

For predictor-side Kp, `kp_asof(q)` returns the most recent canonical 3-hour interval satisfying:

```text
interval_end <= q
```

Primary Kp lag features:

```text
Kp_lag_1h(t)  = kp_asof(t - 1h)
Kp_lag_3h(t)  = kp_asof(t - 3h)
Kp_lag_6h(t)  = kp_asof(t - 6h)
Kp_lag_12h(t) = kp_asof(t - 12h)
Kp_lag_24h(t) = kp_asof(t - 24h)
```

Kp used for target and event truth remains retrospective historical ground truth.

### 2.4 AE and Dst Availability

AE and Dst are retained in raw OMNI ingestion but excluded from the primary causal feature set because the retrospective historical series cannot be shown to consistently equal values available at historical prediction time.

```text
raw ingestion:                 allowed
exploratory analysis:          allowed
primary causal feature set:    excluded
```

### 2.5 CME Availability and Primary-Feature Policy

Phase 0.3 completed the historical availability audit for SOHO/LASCO CDAW CME information.

The audit established a distinction between:

```text
measurement causality
candidate-event causality
```

Timestamped LASCO height-time measurements are sufficiently complete and timely to support causal kinematic reconstruction once a CME candidate is defined.

However, the CDAW CME event universe is manually curated and retrospectively revised. A uniform historical per-event availability or publication rule could not be established across the complete 1996–2025 experiment.

Therefore, CDAW/LASCO CME-derived predictors are excluded from the primary causal feature set.

```text
CDAW raw/research access:             allowed
CDAW source auditing:                 allowed
retrospective scientific analysis:    allowed
primary causal feature set:           excluded
```

This includes CME counts, time since CME, catalog speed, reconstructed speed, acceleration, width, halo state, mass, and kinetic energy.

The exclusion was frozen before CME feature screening, model training, threshold optimization, or final-test inspection and was not based on predictive performance.

Detailed evidence is recorded in `docs/cme_availability.md`.

### 2.6 Target Definition

Primary target:

```text
y_event(t) = max(Kp[t+1 : t+H]) >= T
```

| Parameter | Value |
|---|---:|
| T | 5 |
| H | 6 h |

All models and baselines use exactly the same target for a given H/T experiment.

### 2.7 Event Definition

For event and target ground truth, Kp is interpreted from its canonical 3-hour interval representation and expanded onto the hourly evaluation timeline when the `Z` rule is evaluated.

| Rule | Value |
|---|---|
| Event start | Start of first Kp interval where `Kp >= T` |
| Event end | Last active storm hour before `Kp < T` for `Z` consecutive valid hourly ground-truth states |
| Z | 6 h |
| Separate events | Independent after at least Z valid consecutive below-threshold hours |

With complete aligned Kp data, `Z = 6h` corresponds to two complete consecutive 3-hour Kp intervals below threshold.

### 2.8 Alert Definition

An hourly alert occurs when:

```text
P(storm within H hours) >= tau
```

Alert-producing timestamps separated by `gap <= C` belong to the same episode; `gap > C` starts a new episode. Primary `C = 3h`.

| Condition | Classification |
|---|---|
| `storm_start - H <= first_alert < storm_start` | Early Detection |
| `storm_start <= first_alert <= storm_end` | Late Detection |
| No valid event association | False Alarm |

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

---

## 3. TEMPORAL SPLITS (PHASE 1)

| Split | Period | Purpose |
|---|---|---|
| Initial Train | 1996–2016 | Screening |
| Validation 1 | 2017–2018 | Screening |
| Train 2 | 1996–2018 | Walk-forward fold 1 |
| Validation 2 | 2019–2020 | Walk-forward fold 1 |
| Train 3 | 1996–2020 | Walk-forward fold 2 |
| Validation 3 | 2021 | Walk-forward fold 2 |
| Final Test | 2022–2025 | ONE TIME, AT THE END |

Rules:

1. Never mix future data into training.
2. The Final Test is used once and nothing is modified afterward.
3. Validation and Final Test periods remain fixed.

The 1996 start was established during Phase 0 before model training. It increases solar-cycle regime coverage while preserving all validation and protected-test periods.

---

## 4. FEATURES (PHASE 1)

### 4.1 Feature Layers

| Layer | Features | Purpose |
|---|---|---|
| Raw | Bz, Bt, V, Density, Pressure, Kp_lag_1h/3h/6h/12h/24h | Physical and geomagnetic history |
| Rolling | mean/min/std over 3/6/12/24h as specified | Averages, minima, variability |
| Persistence | Bz and V threshold durations | Duration of adverse conditions |
| Dynamics | 1h/3h deltas and 3h slopes | Changes and trends |
| Interactions | Bz_neg×V, Bz_neg×Density, Pressure×V | Variable combinations |

AE and Dst remain in raw OMNI ingestion for audit and exploratory analysis but are excluded from the primary causal feature set.

CDAW/LASCO CME data are retained only for audit, methodological evidence, and possible future separately specified research extensions.

### 4.2 Primary Predictor Universe

The primary experimental predictor universe is frozen to:

```text
causally eligible OMNI solar-wind measurements
+
conservative causal Kp history
```

No AE, Dst, or CME-derived variable may enter primary feature screening, model selection, threshold optimization, or Final Test evaluation.

### 4.3 Causal Implementation Rule

```text
period_start = raw_timestamp
period_end = raw_timestamp + 1h
information_cutoff = prediction_time - 1h
eligible = period_end <= information_cutoff
```

Rolling, persistence, dynamic, and interaction features must be constructed only from observations that already satisfy this eligibility rule.

---

## 5. BASELINES (PHASE 2)

All baselines produce predictions for the same target.

| Baseline | Model | Purpose |
|---|---|---|
| B0: Persistence | `1 if kp_asof(t-1h) >= T else 0` | Simple persistence |
| B1: Physical | Bz < -X AND V > Y | Simple physical rule |
| B2: Logistic | Logistic Regression, raw primary features | Linear model |
| B3: ExtraTrees | ExtraTrees, raw primary features, no balancing | Simple non-linear model |

Evaluation: Event Recall, FAR/day, Lead Time.

---

## 6. FEATURE ENGINEERING (PHASE 3)

### 6.1 Screening

| Experiment | Features |
|---|---|
| A | Raw physical + causal Kp lag family |
| B | A + Rolling |
| C | B + Persistence |
| D | C + Dynamics |
| E | D + Interactions |

Train ExtraTrees without balancing on Initial Train, evaluate on Validation 1, and record Event Recall, FAR/day, and PR-AUC.

### 6.2 Confirmation

The 2–3 best feature sets proceed to walk-forward confirmation. Selection is based on consistency across folds and Event Recall subject to:

```text
FAR/day <= 0.2
```

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

For thresholds 0.01–0.99 in steps of 0.01, construct canonical alert episodes and calculate Event Recall and FAR/day.

Select the **lowest threshold** satisfying:

```text
FAR/day <= 0.2
```

If no candidate threshold satisfies the constraint, the model fails the operational constraint; the constraint is not relaxed retrospectively.

Report the global threshold, fold-specific thresholds, and diagnostic stability range corresponding to `0.15 <= FAR/day <= 0.20`. The global OOF-selected threshold remains operational.

---

## 10. HORIZON AND SEVERITY EXPERIMENTS (PHASE 7)

Feature set, model, balancing strategy, hyperparameters, alert rules, and **predictor-source universe** remain frozen.

Only H, T, and the OOF-recalibrated threshold change.

Pre-specified combinations:

```text
H = 3, 6, 12, 24 for T = 5
H = 6 for T = 6 and T = 7
```

Excluded Phase 0 sources (AE, Dst, CME) are not introduced during these experiments.

---

## 11. FINAL TEST VALIDATION (PHASE 8)

Before touching the protected 2022–2025 Final Test, the following must be frozen:

```text
primary predictor-source universe
feature set
model family
hyperparameters
imbalance strategy
alert construction
OOF threshold-selection procedure
operational threshold
```

The Final Test is used once. No methodological decision may be modified afterward based on its results.

---

## 12. INTERPRETABILITY AND SCIENTIFIC AUDIT (PHASE 9)

After the protected evaluation:

- report Event Recall, FAR/day, lead-time distributions, late detections, precision, PR-AUC, and calibration;
- inspect SHAP/feature importance for scientific interpretation;
- analyze representative detected, missed, and false-alarm cases;
- evaluate performance across temporal/solar-cycle regimes;
- document limitations without altering the frozen experiment.

Interpretability is diagnostic and explanatory; it must not be used to retroactively tune the protected experiment.

---

## 13. REPRODUCIBILITY AND GOVERNANCE

The repository must preserve:

- protocol versions and amendment history;
- source-semantics documentation;
- Data Contract;
- event and alert definitions;
- source loaders;
- causality and integrity tests;
- configuration;
- deterministic split definitions;
- audit scripts and results supporting source-admission/exclusion decisions.

Historical protocol versions remain unchanged.

### 13.1 Phase 0.3 Decision Record — CME

CME predictors were planned conditionally in earlier protocol versions.

Phase 0.3 investigated the CDAW retrospective CME catalog, historical operational alert products, LASCO Halo Mail, CDAW Version 1 and Version 2 semantics, timestamped LASCO height-time (`.yht`) observations, and cross-era/full-archive `.yht` availability and quality.

The full audit demonstrated that CME measurement-level causal reconstruction is technically feasible, but candidate-event historical availability cannot be established uniformly enough to satisfy the project's primary causal standard.

Accordingly, CME was excluded before model training.

This is a source-availability correction, not a performance-based feature-selection decision.

---

## 14. FUTURE CME RESEARCH

A future protocol may revisit CME remote-sensing information only if a candidate-event source with defensible historical availability is established, or if a frozen automated CME detector is applied causally to historical timestamped imagery.

Such an experiment is outside Version 1.3 and must not alter the present model-development or Final Test procedure.

---

## 15. FROZEN PRIMARY CONFIGURATION

```text
Primary storm threshold T:          5
Primary forecast horizon H:         6 h
Event termination/separation Z:     6 h
Alert episode gap C:                3 h
Maximum FAR/day:                    0.2
Historical development start:       1996
Protected Final Test:               2022–2025
Primary predictor sources:          OMNI solar wind + causal Kp
Excluded primary sources:           AE, Dst, CDAW/LASCO CME
```

These rules may not be changed based on model or Final Test performance.
