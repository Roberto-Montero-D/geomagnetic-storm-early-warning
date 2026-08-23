# Phase 3 Feature-Selection Freeze

**Status:** Frozen after the official walk-forward confirmation run.

## Selected experiment

Phase 3 selects **Experiment A**, containing **10 features**.

The selection is final for subsequent phases. Later modeling phases must import
the frozen Phase 3 feature contract and must not repeat feature selection using
Validation 2, Validation 3, or the protected Final Test.

## Confirmation evidence

The official confirmation evaluated only the candidates that had already
advanced from screening: A, E, and C.

| Candidate | WF1 Event Recall | WF1 FAR/day | WF2 Event Recall | WF2 FAR/day |
|---|---:|---:|---:|---:|
| A | 0.5333 | 0.1789 | 0.7500 | 0.1993 |
| E | 0.4667 | 0.1748 | 0.6875 | 0.1993 |
| C | 0.3333 | 0.1995 | 0.7500 | 0.1680 |

All three candidates were operationally feasible in both confirmation folds
under the frozen FAR/day <= 0.2 constraint.

The frozen ranking prioritized:

1. feasibility in both confirmation folds;
2. highest worst-fold Event Recall;
3. highest mean Event Recall;
4. highest mean PR-AUC;
5. lowest mean FAR/day;
6. frozen candidate order / smaller-set tie resolution.

Under that rule, A ranks first because its worst-fold Event Recall is 0.5333,
higher than E (0.4667) and C (0.3333).

## Scientific interpretation

The larger feature sets produced stronger PR-AUC in some comparisons, but they
did not improve the primary operational robustness criterion. Experiment A
therefore remains selected under the precommitted protocol.

This document records a development-only model-selection decision. It does not
contain or authorize access to protected Final Test performance.
