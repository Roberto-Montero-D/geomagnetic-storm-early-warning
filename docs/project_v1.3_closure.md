# Protocol v1.3 — Project Closure

**Status:** COMPLETE / CLOSED  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Phases:** 0–9 complete  
**Protected Final Test:** consumed once  
**Future confirmatory use of 2022–2025:** forbidden

## Closure Statement

Protocol v1.3 is formally complete.

The project progressed from causal source/timestamp semantics and leakage controls through dataset construction, baselines, feature screening, imbalance experiments, model selection, OOF operational-threshold selection, pre-authorized horizon/severity diagnostics, one-time protected Final Test evaluation, and post-hoc scientific interpretation.

No additional experiment is required to complete v1.3.

## Frozen Primary System

```text
T                      = 5
H                      = 6 h
Z                      = 6 h
C                      = 3 h
maximum FAR/day        = 0.2
information cutoff     = t - 1 h
selected features      = Phase 3 Experiment A
selected feature count = 10
imbalance strategy     = none
model                  = lightgbm_lr0.1_leaves127
operational threshold  = 0.10
```

## Protected Final Test

The protected 2022–2025 interval was executed once in Phase 8 using the unchanged frozen primary configuration.

**Execution commit:** `8c773b1804feabb5cbc1c8dcc08c5340fb20c236`

```text
Event Recall             = 0.5430463576
Detected events          = 82 / 151
FAR/day                  = 0.3680874062
Median early lead time   = 3.0 h
PR-AUC                   = 0.4959636205
ROC-AUC                  = 0.8825873610
Brier score              = 0.0396522111
```

The pre-specified `FAR/day <= 0.20` requirement did not generalize to the protected Final Test. The confirmatory result was not repaired by changing the threshold, model, features, task, or event/alert definitions.

## Scientific Interpretation

The v1.3 result supports a narrower claim than production readiness: the selected predictor/model retained meaningful out-of-sample probability discrimination, but its development-selected operating point did not maintain the required false-alarm burden and detected 54.3% of canonical storm events in 2022–2025.

Phase 9 subsequently performed explanatory post-hoc diagnostics. Those analyses identified temporal/regime-dependent failure behavior and provide hypotheses for a successor system, but they do not alter the Phase 8 result.

## Permanent No-Retuning Boundary

The Phase 8 outcome may not be used within v1.3 to retrospectively change:

- Phase 3 selected features;
- Phase 4 imbalance strategy;
- Phase 5 model family or hyperparameters;
- Phase 6 operational threshold;
- primary `T` or `H`;
- event semantics;
- alert semantics.

The 2022–2025 interval is consumed and is no longer an unseen confirmatory set.

## Reproducibility Record

The repository retains:

- the frozen protocol and phase contracts;
- causal/leakage and temporal-isolation tests;
- model/threshold freeze logic;
- Phase 8 execution commit and environment lock;
- official Phase 8 result and closure documents;
- Phase 9 diagnostic implementation and closure;
- a compact Phase 8 archival summary under `artifacts/phase8_final/`;
- continuous test execution through `.github/workflows/tests.yml`.

The complete local Phase 8 generated artifacts were not recreated during repository closure. No cryptographic hashes are asserted where the original bytes were unavailable to this cleanup.

## Successor Work

Any improved system informed by Phase 8 or Phase 9 must begin under a new protocol/version. Phase 9 findings may be treated as hypothesis-generating evidence, not as permission to relabel a post-hoc improvement as the v1.3 confirmatory result.

A successor confirmatory experiment requires a new unseen evaluation set.

## Final Status

```text
Protocol v1.3              CLOSED
Phases 0–9                 COMPLETE
Protected Final Test       CONSUMED ONCE
Confirmatory retuning      FORBIDDEN
Repository documentation   CLOSED
Next scientific work       NEW PROTOCOL / NEW UNSEEN TEST
```
