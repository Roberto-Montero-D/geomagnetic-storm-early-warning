# Phase 5 — Model Selection Results

**Status:** OFFICIAL DEVELOPMENT-ONLY RESULTS  
**Contract:** `docs/phase5_model_selection_contract.md`  
**Protected Final Test:** NOT USED

## Screening

Phase 5 screened exactly 27 frozen configurations: nine ExtraTrees, nine LightGBM, and nine XGBoost configurations.

Family winners:

| Family | Advancing configuration |
|---|---|
| ExtraTrees | `extratrees_n100_dnone` |
| LightGBM | `lightgbm_lr0.1_leaves127` |
| XGBoost | `xgboost_lr0.1_d9` |

## Walk-Forward Confirmation

| Configuration | Fold | Threshold | Event Recall | FAR/day | PR-AUC |
|---|---|---:|---:|---:|---:|
| `extratrees_n100_dnone` | WF1 | 0.11 | 0.6000 | 0.1926 | 0.2459 |
| `extratrees_n100_dnone` | WF2 | 0.17 | 0.6875 | 0.1993 | 0.2968 |
| `lightgbm_lr0.1_leaves127` | WF1 | 0.07 | 0.7333 | 0.1899 | 0.2532 |
| `lightgbm_lr0.1_leaves127` | WF2 | 0.16 | 0.7500 | 0.1936 | 0.3052 |
| `xgboost_lr0.1_d9` | WF1 | 0.06 | 0.6000 | 0.1871 | 0.2913 |
| `xgboost_lr0.1_d9` | WF2 | 0.11 | 0.6875 | 0.1907 | 0.2981 |

All candidates were feasible in both folds.

## Frozen Ranking

Precommitted order: feasibility in both folds; worst-fold Event Recall; mean Event Recall; mean PR-AUC; mean FAR/day; frozen candidate order.

| Rank | Configuration | Worst recall | Mean recall | Mean PR-AUC | Mean FAR/day |
|---:|---|---:|---:|---:|---:|
| 1 | `lightgbm_lr0.1_leaves127` | 0.7333 | 0.7417 | 0.2792 | 0.1917 |
| 2 | `xgboost_lr0.1_d9` | 0.6000 | 0.6438 | 0.2947 | 0.1889 |
| 3 | `extratrees_n100_dnone` | 0.6000 | 0.6438 | 0.2713 | 0.1960 |

## Frozen Selection

```text
family        = LightGBM
configuration = lightgbm_lr0.1_leaves127
learning_rate = 0.1
num_leaves    = 127
random_state  = 42
imbalance     = none
feature_set   = Phase 3 Experiment A
```

LightGBM wins because the frozen ranking prioritizes temporal Event Recall robustness above PR-AUC. XGBoost's higher mean PR-AUC therefore does not override the selection.

Stacking was not used to reopen the individual-model selection.

## Audit Trail

```text
results/phase5/screening/
results/phase5/confirmation/
```

The protected 2022–2025 Final Test was not used. Phase 6 receives exactly the frozen LightGBM configuration and may not repeat model selection.
