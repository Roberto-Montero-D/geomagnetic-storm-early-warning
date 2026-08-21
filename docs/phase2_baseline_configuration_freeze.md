# Phase 2 Baseline Configuration Freeze

**Status:** Frozen before official Phase 2 baseline evaluation.

This clarification resolves two omissions identified during implementation. No
B1/B3 empirical baseline performance was used to select these values.

## B1 Physical

Official primary configuration:

- `Bz < -5 nT`
- `V > 500 km/s`
- both inequalities remain strict.

Rationale: this is a simple, interpretable moderate-disturbance physical
screening rule using round domain-scale values already represented in the
project's physical persistence thresholds. It is a baseline, not an optimized
coupling function.

## B3 ExtraTrees

Official primary configuration:

- `n_estimators = 100`
- `max_depth = 10`
- `class_weight = None`
- `random_state = 42`

Rationale: use the smallest tree count and shallowest finite depth already
predeclared in the protocol's ExtraTrees candidate space. This makes B3 a
deliberately simple, reproducible baseline rather than selecting a stronger
configuration from observed validation performance.

## Freeze rule

These primary values must not be changed based on Phase 2 development results.
Alternative values, if studied later, belong to the protocol's model-selection
phases and must not retroactively replace the official B1/B3 baseline results.

The protected 2022-2025 Final Test remains untouched.
