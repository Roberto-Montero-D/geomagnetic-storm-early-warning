# Phase 4 — Frozen Imbalance Decision

**Status:** Frozen after the precommitted screening and walk-forward confirmation procedure.

## Selected strategy

`none`

Operational meaning:

- no undersampling;
- no SMOTE;
- no Borderline-SMOTE;
- no SMOTE-ENN;
- no positive-class weighting;
- model training continues on the original eligible training rows.

This decision applies to subsequent phases unless the Master Protocol explicitly
defines a separate experiment in which imbalance treatment is itself a new,
precommitted experimental factor.

## Decision trail

The Phase 4 screening evaluated the 17 configurations frozen before empirical
screening. The three advancing configurations were:

1. `undersample_10_to_1`
2. `none`
3. `class_weight_1`

These candidates were then evaluated on the two frozen confirmation folds:

- WF1: development data through Validation 1 -> Validation 2
- WF2: development data through Validation 2 -> Validation 3

The confirmation ranking rule was frozen before those results were observed:

1. feasible in both confirmation folds;
2. highest minimum (worst-fold) Event Recall;
3. highest mean Event Recall;
4. highest mean PR-AUC;
5. lowest mean FAR/day;
6. frozen candidate order.

All three candidates were feasible in both folds and had the same minimum Event
Recall and mean Event Recall. `none` and `class_weight_1` also had identical mean
PR-AUC and FAR/day, as expected from their equivalent unit weighting. They
ranked ahead of `undersample_10_to_1` on mean PR-AUC. The frozen-order tie-break
therefore selected `none`.

## Scientific interpretation

Within the frozen Phase 4 search space and operational constraint, the tested
imbalance interventions did not demonstrate a temporally robust improvement
over untreated training.

This is a negative experimental result, not permission to reopen Phase 4.
Later model results must not be used to retroactively select a different
resampling or class-weighting strategy.

## Canonical implementation handoff

Subsequent code should import:

```python
from src.imbalance.freeze import PHASE4_FROZEN_DECISION
```

The canonical values are:

```text
experiment      = none
use_resampling  = False
class_weight    = None
```

The empirical CSV files under `results/phase4/` remain the audit trail. This
document and `src/imbalance/freeze.py` encode the frozen methodological decision
for downstream use.
