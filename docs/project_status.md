# Project Status Ledger

**Canonical implementation status:** Protocol v1.3 complete — Phases 0–9 closed  
**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Protected Final Test:** CONSUMED ONCE / CLOSED  
**Phase 9:** POST-HOC DIAGNOSTICS CLOSED

This is the canonical current-status ledger. Historical protocol, contract, decision, and phase-closure documents remain immutable historical records unless a factual repository-reference correction is required.

| Phase | Scope | Status | Frozen handoff / result |
|---|---|---|---|
| 0 | Causality and temporal semantics | COMPLETE | 93-feature causal universe |
| 1 | Dataset, row status, temporal splits | COMPLETE | development folds + protected Final Test |
| 2 | Baselines/evaluation | COMPLETE | canonical operational evaluator |
| 3 | Feature screening | COMPLETE / FROZEN | Experiment A — 10 raw features |
| 4 | Imbalance experiments | COMPLETE / FROZEN | `none` |
| 5 | Model selection | COMPLETE / FROZEN | `lightgbm_lr0.1_leaves127` |
| 6 | OOF threshold selection | COMPLETE / FROZEN | global `tau=0.10` |
| 7 | Horizon/severity experiments | COMPLETE / FROZEN | primary `t5_h6` unchanged |
| 8 | Protected Final Test | COMPLETE / CONSUMED ONCE | official out-of-sample result |
| 9 | Interpretation/scientific audit | COMPLETE / CLOSED | post-hoc diagnostics only |

## Frozen Primary Configuration

```text
T = 5
H = 6 h
Z = 6 h
C = 3 h
maximum FAR/day = 0.2
information cutoff = t - 1h

causal universe = 93 features
selected inputs = Phase 3 Experiment A = 10 raw features
imbalance       = none
model           = lightgbm_lr0.1_leaves127
threshold       = 0.10
```

## Development Freeze

Phase 5 selected `lightgbm_lr0.1_leaves127`. Phase 6 froze `tau=0.10` because `tau=0.09` produced FAR/day `0.20036` and was infeasible while `tau=0.10` produced FAR/day `0.18552` and was feasible.

Phase 7 executed the six pre-authorized truth configurations without changing the frozen predictor/model stack. The `t5_h6` positive control reproduced Phase 6 OOF probabilities exactly, and the primary task remained `T=5, H=6 h`.

## Protected Final Test — Official Phase 8 Result

The protected interval was 2022–2025 and was scored once using the unchanged frozen primary system.

**Execution commit:** `8c773b1804feabb5cbc1c8dcc08c5340fb20c236`

```text
Event Recall             = 0.5430463576
Detected events          = 82 / 151
FAR/day                  = 0.3680874062
Median early lead time   = 3.0 h
Alert episodes           = 611
False-alarm episodes     = 525
Early-detection episodes = 41
Late-detection episodes  = 45
PR-AUC                   = 0.4959636205
ROC-AUC                  = 0.8825873610
Brier score              = 0.0396522111
```

The frozen operational requirement `FAR/day <= 0.20` was **not met** on the protected Final Test. No post-hoc threshold substitution or other retuning is part of the Phase 8 result.

The official interpretation is recorded in `docs/phase8_results.md` and `docs/phase8_closure.md`.

## Phase 9 Closure

Phase 9 performed only post-hoc scientific diagnostics. It examined:

- operational and temporal performance;
- physical error regimes;
- event context and recurrence;
- pre-onset physical state;
- signal timing relative to event onset.

These analyses explain failure modes but do not alter the frozen Phase 8 confirmatory result. Model-internal SHAP attribution was not reconstructed by refitting the protected estimator; that limitation is documented in `docs/phase9_closure.md`.

## Reproducibility Status

The repository contains dedicated test coverage for causal timestamp cutoffs, future-mutation leakage, target/event/alert definitions, temporal split isolation, model and threshold freezes, Phase 8 scoring/provenance, and Phase 9 diagnostics.

Run:

```bash
python -m pytest -q
```

The Phase 8 environment is represented by `requirements-lock-phase8.txt`, the official execution commit is recorded above, and the repository closure includes a compact archival summary under `artifacts/phase8_final/`.

## Protocol v1.3 Closure Rule

Protocol v1.3 is complete. The protected Final Test is consumed and may not be reused as a new confirmatory set.

No v1.3 work may use the protected result to retroactively change:

```text
features
imbalance strategy
model or hyperparameters
operational threshold
primary T or H
event definition
alert definition
```

Any improved successor system informed by Phase 8 or Phase 9 must be declared as a new protocol/version and requires a new unseen confirmatory evaluation set.

See `docs/project_v1.3_closure.md` for the repository-level closure record.