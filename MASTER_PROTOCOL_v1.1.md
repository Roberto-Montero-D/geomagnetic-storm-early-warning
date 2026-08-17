# MASTER PROTOCOL
## Geomagnetic Storm Early Warning System

**Version:** 1.1 (Final — Frozen)
**Date:** August 17, 2026
**Status:** Frozen. No methodological decisions will be modified based on results.

---

## 1. PHILOSOPHY AND OBJECTIVE

### 1.1 Core Philosophy

> This project builds an early warning system for geomagnetic storms. It is **not** a Kp classifier. It is a system that must be causally correct, operationally useful, scientifically interpretable, and temporally robust.

### 1.2 Question the System Answers

> "Will there be a geomagnetic storm (Kp >= T) within the next H hours?"

### 1.3 Definition of Success

> "The system detects X% of events with a median lead time of Y hours, producing Z false alarms per day."

### 1.4 Guiding Principles

1. **Causality:** Only information available in real-time is used.
2. **No leakage:** All features are audited with a Data Contract.
3. **No data snooping:** Decisions are made before seeing results.
4. **Temporal robustness:** Walk-forward validation, not a single partition.
5. **Interpretability:** SHAP and case studies explain predictions.

### 1.5 Freeze Rule

> Once implementation begins, T, H, Z, C, event definition, primary metric, splits, etc. are **not** modified based on results. Only bugs or inconsistencies with the Data Contract may be corrected.

---

## 2. FUNDAMENTAL DEFINITIONS (PHASE 0)

### 2.1 Timestamp and Availability Convention

**Fundamental rule:**

> For a prediction with timestamp `t`, the latest allowed OMNI observation is the hourly observation whose period **ends** at `t-1h`. No observation corresponding to the interval `(t-1h, t]` may be used.

**In code:**

`prediction_time = t`
`latest_allowed_observation = t - 1h`  # End of the last allowed period

**Example:**

- prediction_time = 14:00
- latest_allowed_observation = 13:00
- Allowed OMNI periods: All periods ending <= 13:00
- NOT allowed OMNI periods: 13:00-14:00 (ends at 14:00)

**Mandatory verification:** Before building features, document whether OMNI uses timestamps at the **start** or **end** of the period.

### 2.2 Data Contract

| Feature | Source | Dataset Timestamp | Period Represented | Usable for prediction at t? |
|---------|--------|-------------------|-------------------|----------------------------|
| Bz | OMNI | END of period | [timestamp-1h, timestamp] | Yes, if timestamp <= t-1h |
| Bt | OMNI | END of period | [timestamp-1h, timestamp] | Yes, if timestamp <= t-1h |
| V | OMNI | END of period | [timestamp-1h, timestamp] | Yes, if timestamp <= t-1h |
| Density | OMNI | END of period | [timestamp-1h, timestamp] | Yes, if timestamp <= t-1h |
| Pressure | OMNI | END of period | [timestamp-1h, timestamp] | Yes, if timestamp <= t-1h |
| AE | OMNI | END of period | [timestamp-1h, timestamp] | Yes, if timestamp <= t-1h |
| Dst | OMNI | END of period | [timestamp-1h, timestamp] | Yes, if timestamp <= t-1h |
| Kp | Kp index | Exact hour | Hourly value | Yes, if timestamp <= t-1h |
| CME | Catalog | Variable | Point event | Only if cme_available_at_t == True |

### 2.3 Specific Rule for CME (CRITICAL)

**The rule is:**

> `cme_information_available_at_t == True`

**Usable CME characteristics (only if available at t):**

- `hours_since_cme_observation`
- `hours_until_cme_eta` (only if ETA available)
- `cme_speed_if_available`
- `cme_energy_if_available`

**Exclusion rule:**

> CMEs without `cme_available_at_t == True` are **not used** in training or evaluation.

### 2.4 Target Definition

**Primary target:**

> `y_event(t) = max(Kp[t+1 : t+H]) >= T`

**Fixed parameters (initial):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| T | 5 | Kp threshold for storm |
| H | 6 | Prediction horizon in hours |

**Parameters to experiment with (Phase 7):**

| Parameter | Values |
|-----------|--------|
| T | 5, 6, 7 |
| H | 3, 6, 12, 24 |

**Rule:** All models and baselines use exactly the same target.

### 2.5 Event Definition (Storm Episodes)

**Rules (fixed before training):**

| Rule | Value | Description |
|------|-------|-------------|
| Event start (storm_start) | First hour where Kp >= T | - |
| Event end (storm_end) | Last hour before Kp < T for Z consecutive hours | - |
| Z | 6 hours | Cooldown to separate events |
| Separate events | Independent | If Kp remains < T for >= Z hours between them |

**Counting rule:**

> Each real event is counted **once**, regardless of the number of alert episodes that overlap it.

**Deliverable:** Function `identify_events(kp_series, threshold=T, cooldown=Z)`.

### 2.6 Alert and Alert Episode Definition

**Rules (fixed before training):**

| Rule | Value | Description |
|------|-------|-------------|
| Hourly alert | Occurs when P(storm) >= tau | tau = operational threshold |
| Alert episode | Group of consecutive alerts | - |
| C | 3 hours | Cooldown between episodes |

**Counting rule:**

> Each alert episode is associated with at most one real event.

**Alert episode classification:**

| Condition | Classification |
|-----------|---------------|
| first_alert < storm_start - H | False Alarm |
| storm_start - H <= first_alert < storm_start | Early Detection |
| storm_start <= first_alert <= storm_end | Late Detection |
| No overlap with any event | False Alarm |

**Correct detection:** Early Detection or Late Detection.

**Deliverable:** Function `identify_alerts(prob_series, threshold=tau, cooldown=C, H=H)`.

### 2.7 Official Metrics

| Metric | Definition | Calculation |
|--------|------------|-------------|
| Event Recall | Number of Detected events / Number of Real events | Event-level bootstrap for CI |
| FAR/day | Number of False alert episodes / Number of Total days | Temporal block bootstrap |
| Lead Time | storm_start - first_alert | Only for Early Detections |
| Late Detection Rate | Number of Late Detections / Number of Detected events | Proportion |
| Precision | Number of Correct episodes / Number of Total episodes | Secondary |
| PR-AUC | Area under Precision-Recall curve | Secondary |
| Reliability | Calibration curve, Brier Score | Secondary |

**Primary operational constraint:**

> `FAR/day <= 0.2`

---

## 3. TEMPORAL SPLITS (PHASE 1)

### 3.1 Split Structure

| Split | Period | Purpose |
|-------|--------|---------|
| Initial Train | 2008-2016 | Screening |
| Validation 1 | 2017-2018 | Screening |
| Train 2 | 2008-2018 | Walk-forward fold 1 |
| Validation 2 | 2019-2020 | Walk-forward fold 1 |
| Train 3 | 2008-2020 | Walk-forward fold 2 |
| Validation 3 | 2021 | Walk-forward fold 2 |
| Final Test | 2022-2025 | ONE TIME, AT THE END |

### 3.2 Split Rules

1. **Never** mix future data into training.
2. The **Final Test** is used **once** and nothing is modified afterward.
3. Splits are **fixed** and not changed based on results.

### 3.3 Solar-Cycle Context and Screening Limitation

The temporal splits are fixed independently of the observed model results. Because geomagnetic activity is strongly time-dependent and varies with the solar cycle, the protocol will document the solar-cycle context of each split in the final report.

In particular:

- **Initial Train (2008-2016):** spans the declining phase of Solar Cycle 23/early Solar Cycle 24 and the rise toward the Solar Cycle 24 maximum.
- **Validation 1 (2017-2018):** spans the late phase of Solar Cycle 24 and the transition toward Solar Cycle 25.
- **Validation 2 (2019-2020):** spans the Solar Cycle 24 minimum and the beginning of Solar Cycle 25.
- **Validation 3 (2021):** covers the rising phase of Solar Cycle 25.
- **Final Test (2022-2025):** covers the strong rise and maximum of Solar Cycle 25.

The use of Validation 1 (2017-2018) for initial screening is therefore recognized as a potential source of selection sensitivity to a particular phase of the solar cycle. This is **not corrected by changing the predefined split**. Instead, robustness to temporal regime changes is assessed through the subsequent walk-forward confirmation folds.

The final report will explicitly state this limitation and compare the selected configuration across all walk-forward validation periods.

---

## 4. FEATURES (PHASE 1)

### 4.1 Feature Layers

**All features are built using `latest_allowed_observation = t - 1h`.**

| Layer | Features | Purpose |
|-------|----------|---------|
| Layer 1: Raw | Bz, Bt, V, Density, Pressure, AE, Dst, Kp(t-1h), Kp(t-3h), Kp(t-6h), Kp(t-12h), Kp(t-24h) | Physical baseline |
| Layer 2: Rolling | rolling_mean_3/6/12/24h, rolling_min_3/6/12/24h (Bz), rolling_std_3/6/12/24h | Averages, minima, variability |
| Layer 3: Persistence | Bz_negative_less_than_-5_duration, Bz_negative_less_than_-10_duration, Bz_negative_less_than_-15_duration, V_high_greater_than_500_duration, V_high_greater_than_600_duration | Duration of adverse conditions |
| Layer 4: Dynamics | delta_Bz_1h, delta_Bz_3h, delta_V_1h, delta_V_3h, slope_Bz_3h, slope_V_3h | Changes and trends |
| Layer 5: CME | hours_since_last_CME, days_since_last_CME, CME_count_last_24h, CME_count_last_48h, CME_count_last_72h, max_CME_speed_last_24h, max_CME_speed_last_48h, max_CME_speed_last_72h, hours_until_CME_eta | Recent CME activity |
| Layer 6: Interactions | Bz_neg_multiply_V, Bz_neg_multiply_Density, Pressure_multiply_V | Variable combinations |

### 4.2 Code Implementation

```
# Timestamp rule
latest_allowed_observation = t - 1h

# Feature construction
df_omni_filtered = df_omni[df_omni.index <= latest_allowed_observation]

# Rolling with shift(1) to ensure causality
rolling_Bz_3h = df_omni_filtered['Bz'].rolling('3h').mean().shift(1)
```

---

## 5. BASELINES (PHASE 2)

**Rule:** All baselines produce `y_hat_event(t)` for comparison with `y_event(t) = max(Kp[t+1:t+H]) >= T`.

| Baseline | Model | Purpose |
|----------|-------|---------|
| B0: Persistence | y_hat(t) = 1 if Kp(t-1h) >= T | Simple persistence |
| B1: Physical | Bz < -X AND V > Y (X, Y optimized) | Simple physical rule |
| B2: Logistic | Logistic Regression (raw features) | Linear model |
| B3: ExtraTrees | ExtraTrees (raw features, no balancing) | Simple non-linear model |

**Evaluation:** Event Recall, FAR/day, Lead Time.

---

## 6. FEATURE ENGINEERING (PHASE 3)

### 6.1 Screening (Validation 1)

| Experiment | Features | Purpose |
|------------|----------|---------|
| A | Raw physical + basic Kp | Baseline |
| B | A + Rolling | Do averages/min/max help? |
| C | B + Persistence | Does duration capture value? |
| D | C + Dynamics | Do abrupt changes matter? |
| E | D + CME temporal | Does recent CME activity add value? |
| F | E + Interactions | Are combinations important? |

**Procedure:**

1. Train ExtraTrees (no balancing) on Initial Train.
2. Evaluate on Validation 1.
3. Record Event Recall, FAR/day, PR-AUC.

### 6.2 Confirmation (Walk-Forward)

- The 2-3 best feature sets proceed to walk-forward.
- Criteria: Consistency across folds, best Event Recall with FAR/day <= 0.2.

**Screening limitation:** Initial feature screening is performed on Validation 1 (2017-2018), so the first selection step may be sensitive to the physical regime represented by that period. This is intentional and fixed by the protocol. Walk-forward confirmation is the safeguard against selecting a configuration that performs well only in that screening period.

The final report will document this limitation and report fold-level results rather than relying only on the Validation 1 ranking.

---

## 7. IMBALANCE HANDLING (PHASE 4)

### 7.1 Screening (Validation 1)

| Experiment | Strategy | Parameters |
|------------|----------|------------|
| 1 | No balancing | Threshold optimized |
| 2 | Class weighting | pos_weight = 1, 3, 5, 10, 20, 50 |
| 3 | Random undersampling | Ratio 10:1, 5:1, 2:1 |
| 4 | SMOTE | k=3, 5, 7 |
| 5 | Borderline-SMOTE | k=3, 5, 7 |
| 6 | SMOTE-ENN | - |

**Procedure:**

1. Use best feature set from Phase 3.
2. Train ExtraTrees on Initial Train.
3. Evaluate on Validation 1.
4. Optimize threshold for Event Recall with FAR/day <= 0.2.

### 7.2 Confirmation (Walk-Forward)

- Top 5 strategies proceed to walk-forward.
- Additional analysis: Calibration (curve, Brier Score), threshold stability.

**Screening limitation:** The initial balancing strategy selection is performed on Validation 1 (2017-2018). As with feature screening, this can introduce sensitivity to the temporal regime represented by that period. Walk-forward confirmation is therefore required before the strategy is considered final, and fold-level performance and threshold stability will be reported.

---

## 8. MODEL SELECTION (PHASE 5)

### 8.1 Comparison (Walk-Forward)

| Model | Hyperparameters |
|-------|-----------------|
| ExtraTrees | n_estimators = 100, 200, 500; max_depth = 10, 20, None |
| LightGBM | learning_rate = 0.01, 0.05, 0.1; num_leaves = 31, 63, 127 |
| XGBoost | learning_rate = 0.01, 0.05, 0.1; max_depth = 3, 6, 9 |

**Stacking:** Only if ExtraTrees and LightGBM show complementary errors.

---

## 9. OPERATIONAL OPTIMIZATION (PHASE 6)

### 9.1 Out-of-Fold (OOF) Predictions

**Procedure:**

1. For each walk-forward fold, train on corresponding Train set.
2. Generate predictions on corresponding Validation set.
3. Store OOF predictions **preserving temporal order**.
4. Retain: `timestamp`, `probability`, `target`, `storm_id`, `fold`.

### 9.2 Global Threshold Selection

**On temporally ordered OOF predictions:**

1. Apply `identify_alerts` for each threshold (0.01 to 0.99, step 0.01).
2. Calculate Event Recall and FAR/day.
3. Select the **lowest threshold** that satisfies `FAR/day <= 0.2`.

**Final threshold = global threshold selected on OOF.**

### 9.2.1 Threshold Stability Analysis

Because the global threshold is selected from a finite number of walk-forward folds, the protocol will also quantify threshold stability without changing the primary selection rule.

For each walk-forward fold, the threshold that satisfies `FAR/day <= 0.2` will be recorded. In addition, the analysis will report the range of thresholds whose corresponding operating point has `FAR/day` within the interval **[0.15, 0.20]**.

The following will be reported:

- Global OOF-selected threshold.
- Fold-specific thresholds satisfying `FAR/day <= 0.2`.
- Minimum and maximum fold-specific thresholds.
- Range of thresholds satisfying `0.15 <= FAR/day <= 0.20`.
- Event Recall across that stability range.

This is a **diagnostic stability analysis only**. The final operational threshold remains the pre-specified global OOF threshold and is not replaced by a more favorable value after observing the results.

### 9.3 Operating Points Table

| Max FAR/day | Threshold | Event Recall | Lead Time (median) |
|-------------|-----------|--------------|-------------------|
| 0.05 | ... | ... | ... |
| 0.10 | ... | ... | ... |
| **0.20** | ... | ... | ... |
| 0.50 | ... | ... | ... |
| 1.00 | ... | ... | ... |

---

## 10. HORIZON AND SEVERITY EXPERIMENTS (PHASE 7)

**Frozen:** Feature set, model, balancing, hyperparameters, alert rules.

**Change:** Target (H and T), threshold (recalibrated with OOF protocol).

**Pre-specified limitation:** The feature set, model, and balancing strategy are selected and frozen for the primary task (`T=5`, `H=6h`) before the Phase 7 experiments. They are therefore **not re-optimized separately for each alternative H/T combination**. Phase 7 is intended to measure how the frozen primary system generalizes across horizons and severity thresholds, rather than to identify the optimal model for every H/T configuration.

This limitation will be explicitly reported in the final scientific discussion.

| H | T | Prevalence | Event Recall | FAR/day | Lead Time |
|---|----|------------|--------------|---------|-----------|
| 3h | 5 | ... | ... | ... | ... |
| 6h | 5 | ... | ... | ... | ... |
| 12h | 5 | ... | ... | ... | ... |
| 24h | 5 | ... | ... | ... | ... |
| 6h | 6 | ... | ... | ... | ... |
| 6h | 7 | ... | ... | ... | ... |

---

## 11. FINAL TEST VALIDATION (PHASE 8)

### 11.1 Final Freeze

**Before touching Test, everything is fixed:**

- Feature set
- Model and hyperparameters
- Balancing strategy
- Threshold (from Phase 6)
- `C = 3h`, `Z = 6h`, `H = 6h`, `T = 5`

### 11.2 Execution

1. Retrain with all data up to 2021.
2. **DO NOT TOUCH TEST during training.**
3. Run on Test (2022-2025).
4. Calculate all metrics.

### 11.3 Confidence Intervals

- **Event Recall:** Event-level bootstrap (1000 samples).
- **Lead Time:** Bootstrap on detected events (Early Detections).
- **FAR/day:** Temporal block bootstrap.

### 11.4 Expected Results

```
Test Results (2022-2025):

Event Recall:       X% [95% CI: Y% - Z%]
Precision:          X% [95% CI: Y% - Z%]
FAR/day:            X [95% CI: Y - Z]
Lead Time (median): Xh [95% CI: Yh - Zh]
Late Detection Rate: X%
PR-AUC:             X.XXX
```


---

## 12. ERROR ANALYSIS AND INTERPRETATION (PHASE 9)

### 12.1 False Negatives

For each missed storm:

- Sudden onset?
- Anomalous conditions?
- Unlisted CME?

### 12.2 False Alarms

For each false alarm:

- What conditions produced it?
- Was there a real CME or other physically meaningful precursor?
- Did a real CME occur but fail to produce a storm within the operational horizon?
- Was the alert caused primarily by noisy or ambiguous model evidence?

False alarms will be further categorized for interpretation into:

1. **Physically plausible / CME-associated false alarms:** a real solar transient or adverse solar-wind condition was present, but no qualifying geomagnetic storm occurred within the operational association window.
2. **Model/noise false alarms:** no clearly identifiable physical precursor or adverse condition explains the alert.

This distinction does **not** change the official FAR/day metric. It is used only for scientific error analysis so that operational false alarms are not incorrectly interpreted as evidence that the underlying physical signal had no value.

### 12.3 SHAP / Feature Importance

- Top 10 global features.
- SHAP summary plot.
- Case analysis.

### 12.4 Case Studies

3-5 representative storms:

- Actual Kp vs prediction.
- Alerts and lead time.
- Key features.
- SHAP at alert time.

---

## 13. SCIENTIFIC ABLATION EXPERIMENT (PHASE 10)

### 13.1 Design

| Experiment | Features | Purpose |
|------------|----------|---------|
| M0 | Only Kp(t-1h) | Simple persistence |
| M1 | Kp(t-1h), Kp(t-3h), Kp(t-6h), Kp(t-12h), Kp(t-24h) | Geomagnetic dynamics |
| M2 | Solar wind + CME (temporal) | Solar conditions |
| M3 | M1 + M2 | Combined |

### 13.2 Procedure

1. Each model: same architecture (ExtraTrees + class weighting).
2. Same hyperparameters.
3. Each model: threshold selected with OOF protocol.
4. Training up to 2021.
5. Evaluation on Test **once**.

### 13.3 Pre-registered Statement

> "M0-M3 constitute a pre-specified confirmatory ablation experiment. No feature, hyperparameter, threshold, or model modification will be performed after observing test results."

### 13.4 Expected Results

```
Test Results:

M0 (Simple persistence):  Event Recall = X%, FAR/day = 0.20
M1 (Kp history):          Event Recall = X%, FAR/day = 0.20
M2 (Solar wind + CME):    Event Recall = X%, FAR/day = 0.20
M3 (Combined):            Event Recall = X%, FAR/day = 0.20
```

---

## 14. ESTIMATED TIMELINE

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 0 | 1 week | Data Contract, definitions, functions |
| Phase 1 | 1 week | Dataset built and audited |
| Phase 2 | 1 week | Baselines |
| Phase 3 | 2 weeks | Feature screening + confirmation |
| Phase 4 | 1.5 weeks | Balancing screening + confirmation |
| Phase 5 | 1 week | Model selected |
| Phase 6 | 1 week | OOF threshold, operational table |
| Phase 7 | 1 week | H/T experiments |
| Phase 8 | 1 week | Final Test + CI |
| Phase 9 | 1 week | Error analysis, SHAP |
| Phase 10 | 1 week | M0-M3 experiment |

**Total estimated:** 12.5 - 13.5 weeks (realistic: 14-16 weeks)

---

## 15. KEY DECISIONS - FINAL SUMMARY

| Decision | Value |
|----------|-------|
| Primary target | Event-in-Window, T=5, H=6h |
| Z (event cooldown) | 6h |
| C (alert cooldown) | 3h |
| Max FAR/day | 0.2 |
| Temporal buffer | latest_allowed_observation = t - 1h |
| Final threshold | OOF predictions |
| Event counting | Each event counted once |
| Alert association | Each alert associated with one event |
| Too-early alerts | first_alert < storm_start - H -> False Alarm |

---

## 16. QUESTIONS ANSWERED BY THE FINAL REPORT

The final report will also explicitly document the methodological limitations specified in this protocol, including screening sensitivity to the 2017-2018 validation regime, the fixed-primary-task design of Phase 7, and operational threshold stability.

1. **How many storms do we detect?** -> Event Recall = X% [CI]
2. **With how much advance notice?** -> Lead Time median = Y hours [CI]
3. **How many detections are late?** -> Late Detection Rate = X%
4. **At the cost of how many false alarms?** -> FAR/day = Z [CI]
5. **Which variables are most important?** -> Top 5 features (SHAP)
6. **How does performance vary with horizon/severity?** -> H/T table
7. **How much does each information family contribute?** -> M0-M3 comparison

---

## 17. FREEZE DECLARATION

> **This document constitutes the master protocol of the project. All methodological decisions contained herein are final and will not be modified based on results observed during execution. Any deviation requires documented justification based exclusively on implementation issues or data inconsistencies, not on model performance.**

---

**End of Master Protocol**