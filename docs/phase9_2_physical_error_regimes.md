# Phase 9.2 — Physical Error-Regime Diagnostics

**Status:** POST-HOC / EXPLORATORY  
**Phase 8 result changed:** NO

Phase 9.2 asks whether detected storms, missed storms, and false-alarm episodes
occupy visibly different causal physical states.

The diagnostic feature universe is not selected from Final Test behavior. It is
exactly frozen Phase 3 Experiment A: the 10 raw predictors already selected
before the protected test.

For each protected storm, features are summarized over `[storm_start-6h,
storm_start)`. The storm-start timestamp itself and all later information are
excluded. For false alarms, the feature state is read at the immutable
first-alert timestamp.

Outputs are descriptive median/IQR summaries and per-event/per-false-alarm
snapshots. No hypothesis-test fishing, feature ranking, threshold optimization,
model fitting, or feature selection is authorized.

The purpose is explanation: identify candidate physical regimes associated
with misses and false alarms, especially whether 2025 differs from earlier
Final Test years. Any resulting hypothesis belongs to a future protocol and
requires a new unseen confirmatory evaluation.
