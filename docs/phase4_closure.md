# Phase 4 — Imbalance Strategy Closure

**Status:** CLOSED

Phase 4 is complete. Its purpose was to determine whether a frozen set of
class-imbalance interventions improved the operational early-warning system
under the development-only protocol.

## Frozen outcome

The selected strategy is:

`none`

Downstream meaning:

- no resampling;
- no class weighting;
- retain the original supervised-eligible training distribution.

The canonical machine-readable handoff is
`src/imbalance/freeze.py`.

## Completed decision path

1. The Phase 4 experimental contract was frozen before results.
2. Seventeen imbalance configurations were screened on the frozen initial
   screening fold.
3. The precommitted top three advanced:
   - `undersample_10_to_1`
   - `none`
   - `class_weight_1`
4. Those three were evaluated on WF1 and WF2.
5. All three were operationally feasible in both confirmation folds.
6. The frozen confirmation ranking selected `none`.
7. The decision was encoded in `src/imbalance/freeze.py` and documented in
   `docs/phase4_imbalance_decision.md`.

## Protected-test status

Phase 4 did not use the protected Final Test for screening, threshold selection,
candidate advancement, confirmation, or strategy selection.

The Final Test remains reserved for the protocol-defined final evaluation.

## Closure rule

Phase 4 must not be reopened because a later model, feature set, threshold, or
final-test result appears to favor a different imbalance treatment.

A future imbalance experiment is permissible only if it is explicitly defined
as a new precommitted experiment rather than presented as a revision of this
closed phase.

## Downstream contract

All subsequent phases inherit:

```text
imbalance_experiment = none
use_resampling       = False
class_weight         = None
```

Subsequent model-training code should consume the frozen Phase 4 decision rather
than independently selecting or inferring an imbalance strategy.

## Audit trail

The empirical development artifacts remain under:

- `results/phase4/screening/`
- `results/phase4/confirmation/`

The methodological decision is recorded in:

- `docs/phase4_imbalance_decision.md`
- `src/imbalance/freeze.py`

With this closure, Phase 4 is complete and no additional Phase 4 empirical
evaluation is authorized.
