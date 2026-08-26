# Geomagnetic Storm Early Warning System

**Status:** Protocol Frozen — Phases 0–6 Complete; Phase 7 Next  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Primary Horizon:** 6 hours  
**Primary Storm Threshold:** Kp >= 5  
**Frozen Model:** `lightgbm_lr0.1_leaves127`  
**Frozen Operational Threshold:** `tau = 0.10`

## 1. Project Overview

This repository implements a scientifically controlled early-warning system for geomagnetic storms.

> Given only information available at prediction time `t`, can the system warn that geomagnetic storm conditions will occur within the next `H` hours?

Development decisions are frozen before later results are inspected. The primary predictor universe is causally eligible OMNI solar-wind measurements plus conservative causal Kp history. AE, Dst, and CDAW/LASCO CME-derived predictors remain excluded from the primary causal feature matrix.

The protected 2022–2025 Final Test remains untouched.

## 2. Frozen Primary Configuration

| Parameter | Frozen value |
|---|---:|
| Storm threshold `T` | 5 |
| Forecast horizon `H` | 6 h |
| Event separation / termination `Z` | 6 h |
| Alert episode gap `C` | 3 h |
| Maximum FAR/day | 0.2 |
| Information cutoff | `t - 1h` |
| Canonical prediction grid | `[1996-01-01, 2026-01-01)` hourly |
| Protected Final Test | 2022–2025 |
| Causal feature universe | 93 |
| Selected model-input set | Experiment A — 10 raw features |
| Imbalance strategy | `none` |
| Model | `lightgbm_lr0.1_leaves127` |
| Operational threshold | `0.10` |

## 3. Feature Universe vs. Selected Model Input

Phase 0 froze a **93-feature causal universe**: raw, rolling, persistence, dynamics, and interactions. Phase 1 therefore builds 93 predictors plus the target.

Phase 3 later selected **Experiment A**, the **10-feature raw family**, for downstream primary modeling.

> The canonical dataset contains 93 predictors, while the frozen Phase 4–6 model uses the 10-feature Phase 3 Experiment A subset.

The remaining causal features stay in the dataset/audit infrastructure but are not silently reintroduced into the selected model.

## 4. Canonical Target and Causality

```text
y_event(t) = max(Kp[t+1:t+H]) >= T
T = 5
H = 6h
window = (t, t+H]
```

Despite the historical name `y_event`, this is a future storm-condition target, not an event-onset-only target.

For prediction time `t`:

```text
information_cutoff = t - 1h
maximum_feature_information_time <= information_cutoff
```

Future retrospective Kp may define target/event truth but may never enter the predictor matrix.

## 5. Temporal Validation

| Atomic period | Period |
|---|---|
| Initial Train | 1996–2016 |
| Validation 1 | 2017–2018 |
| Validation 2 | 2019–2020 |
| Validation 3 | 2021 |
| Final Test | 2022–2025 |

```text
screening       1996–2016 -> 2017–2018
walk_forward_1  1996–2018 -> 2019–2020
walk_forward_2  1996–2020 -> 2021
```

Final Test rows are excluded from every development train and validation mask.

## 6. Phase Summary

### Phase 0 — COMPLETE
Frozen causal timestamp/availability semantics, canonical Kp, storm events, alert episodes, 93-feature causal pipeline, target construction, and leakage/temporal-integrity tests.

### Phase 1 — COMPLETE
Frozen canonical hourly grid, row-preserving 93-feature dataset, row status, development folds, and protected Final Test isolation.

### Phase 2 — COMPLETE
Implemented B0 persistence, B1 physical, B2 Logistic Regression, B3 ExtraTrees, and canonical development-only operational evaluation.

### Phase 3 — COMPLETE / FROZEN
Selected **Experiment A — 10 raw features**.

### Phase 4 — COMPLETE / FROZEN
Selected imbalance strategy:

```text
none
```

No resampling and no class weighting.

### Phase 5 — COMPLETE / FROZEN
Screened exactly 27 configurations. Family winners:

```text
ExtraTrees -> extratrees_n100_dnone
LightGBM   -> lightgbm_lr0.1_leaves127
XGBoost    -> xgboost_lr0.1_d9
```

Walk-forward confirmation selected:

```text
lightgbm_lr0.1_leaves127
```

| Fold | Event Recall | FAR/day | PR-AUC |
|---|---:|---:|---:|
| WF1 | 0.7333 | 0.1899 | 0.2532 |
| WF2 | 0.7500 | 0.1936 | 0.3052 |

Aggregate ranking values: worst-fold Event Recall `0.7333`, mean Event Recall `0.7417`, mean PR-AUC `0.2792`, mean FAR/day `0.1917`.

### Phase 6 — COMPLETE / FROZEN
Generated 25,873 development-only OOF predictions from the frozen LightGBM model and swept `tau=0.01..0.99`.

Frozen rule: select the **lowest** threshold satisfying `FAR/day <= 0.2`.

```text
tau = 0.09 -> FAR/day = 0.20036 -> infeasible
tau = 0.10 -> FAR/day = 0.18552 -> feasible
```

Therefore:

```text
Frozen tau       = 0.10
Event Recall     = 21 / 31 = 0.6774
FAR/day          = 0.1855
Alert episodes   = 222
False alarms     = 200
```

Stability thresholds for predefined `0.15 <= FAR/day <= 0.20`:

```text
0.10, 0.11, 0.12, 0.13
```

Fold diagnostic thresholds:

```text
WF1 = 0.07
WF2 = 0.16
```

At `tau=0.13`, observed OOF recall is `22/31 = 0.7097` with FAR/day `0.1568`. This is a diagnostic only; it does not reopen the frozen minimum-feasible-threshold rule.

## 7. Final Test Protection

The 2022–2025 Final Test may not influence feature, imbalance, model, hyperparameter, threshold, or candidate-selection decisions before its protocol-authorized evaluation.

The official Phase 6 summary records:

```text
protected_final_test_scored = false
```

## 8. Current Status

| Phase | Status |
|---|---|
| 0 — Causality/temporal infrastructure | COMPLETE |
| 1 — Dataset/splits | COMPLETE |
| 2 — Baselines | COMPLETE |
| 3 — Feature screening | COMPLETE / FROZEN |
| 4 — Imbalance experiments | COMPLETE / FROZEN |
| 5 — Model selection | COMPLETE / FROZEN |
| 6 — OOF threshold selection | COMPLETE / FROZEN |
| 7 — Horizon/severity experiments | NEXT |
| 8 — Protected Final Test | LOCKED |
| 9 — Interpretation/scientific audit | PENDING |

## 9. Documentation and Results

Current decision records include the Phase 0–4 documents plus:

```text
docs/phase5_model_selection_contract.md
docs/phase5_model_selection_results.md
docs/phase5_closure.md
docs/phase6_threshold_selection_contract.md
docs/phase6_threshold_selection_results.md
docs/phase6_closure.md
docs/project_status.md
```

Historical protocol versions and historical decision documents remain unchanged.

Machine-readable results include:

```text
results/phase5/screening/
results/phase5/confirmation/
results/phase6/threshold_selection/
```

## 10. Reproducibility

Run the complete test suite with:

```bash
python -m pytest -q
```

Phase 6 runner:

```bash
python -m scripts.run_phase6_threshold_selection   --omni-fmt <path-to-omni.fmt>   --omni-lst <path-to-omni.lst>   --output-dir results/phase6/threshold_selection
```

This runner is development-only.

## 11. Next Step

Proceed to **Phase 7 — horizon/severity experiments** under its frozen protocol contract. Phases 3–6 must not be reopened, and the protected Final Test remains locked.
