# Phase 8 — Closure

**Status:** COMPLETE / FINAL TEST CONSUMED  
**Protected Final Test:** SCORED ONCE  
**Execution commit:** `8c773b1804feabb5cbc1c8dcc08c5340fb20c236`  
**Primary configuration changed after test:** NO

Phase 8 is formally closed.

The protected 2022–2025 Final Test was executed once using the unchanged
primary configuration frozen before outcome access.

## Frozen Final Result

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

The frozen `FAR/day <= 0.20` operational requirement was not met.

## No-Retuning Rule

The following are permanently frozen with respect to the Phase 8 confirmatory
result:

```text
Phase 3 selected features
Phase 4 imbalance decision
Phase 5 model family/configuration
Phase 6 operational threshold
primary T
primary H
event definition
alert definition
```

The protected Final Test result must not be used to:

- change `tau=0.10` and then report the new value as the Phase 8 result;
- switch to another Phase 7 horizon/severity task;
- reselect features;
- reweight or resample classes;
- choose another model/hyperparameter configuration;
- alter event or alert semantics;
- rerun the protected test as an improved confirmatory evaluation.

## Authorized Next Work

Phase 9 may perform post-hoc scientific diagnostics.

Authorized diagnostic questions include:

- Did probability calibration drift by year?
- Did target prevalence change across 2022–2025?
- Did false-alarm episodes cluster in particular years?
- How did PR-AUC, ROC-AUC, and Brier score vary by year?
- What is the distribution of early-detection lead times?
- Are false alarms concentrated in particular probability ranges or temporal
  regimes?
- Are there indications of solar-cycle/regime shift?

These analyses are explanatory only.

If a future system version is developed using knowledge from Phase 8, it must
be labeled a new post-Phase-8 protocol/version and requires a new unseen
evaluation set for confirmatory testing.
