# Phase 6 — OOF Operational Threshold Selection Contract

**Status:** FROZEN / EXECUTED  
**Inherited model:** `lightgbm_lr0.1_leaves127`  
**Protected Final Test:** LOCKED

## Purpose and Inheritance

Phase 6 selects the definitive global operational probability threshold using development-only OOF predictions. It does not repeat feature, imbalance, model-family, or hyperparameter selection.

```text
feature set     = Phase 3 Experiment A
imbalance       = none
model           = lightgbm_lr0.1_leaves127
T               = 5
H               = 6 h
Z               = 6 h
C               = 3 h
maximum FAR/day = 0.2
```

## OOF Construction

The frozen model is refit independently for WF1 and WF2. Validation probabilities are concatenated while preserving fold identity. Alert episodes are constructed independently within each fold and may not bridge fold boundaries.

The protected Final Test is excluded.

## Threshold Grid and Metrics

```text
tau = 0.01, 0.02, ..., 0.99
```

Global FAR/day is total false-alarm episodes divided by total valid OOF exposure in days; it is not the arithmetic mean of fold FAR.

Global Event Recall is unique eligible events detected divided by eligible events; it is event-weighted, not the arithmetic mean of fold recall.

## Frozen Selection Rule

> Select the lowest threshold satisfying `global FAR/day <= 0.2`.

The rule does not maximize recall among feasible thresholds.

Fold-specific minimum feasible thresholds are diagnostics only.

The predefined stability diagnostic is:

```text
0.15 <= global FAR/day <= 0.20
```

## Final Test and Required Artifacts

A valid primary Phase 6 run requires:

```text
protected_final_test_scored = false
```

Official artifacts:

```text
oof_predictions.csv
global_threshold_curve.csv
fold_threshold_curves.csv
fold_selected_thresholds.csv
stability_thresholds.csv
phase6_selection_summary.json
```
