# Phase 9 — Post-Hoc Scientific Diagnostics Contract

**Status:** EXPLORATORY / INTERPRETIVE  
**Phase 8 confirmatory result:** FROZEN  
**Threshold optimization:** FORBIDDEN  
**Model retraining:** FORBIDDEN

Phase 9 begins only after the protected Final Test has been consumed and Phase
8 has been closed.

Its purpose is to explain the observed generalization gap without modifying the
official result.

## Frozen Inputs

The initial diagnostic runner consumes only the immutable Phase 8 artifacts:

```text
final_test_predictions.csv
final_test_alert_episodes.csv
final_test_metrics.json
```

It does not refit a model and does not search thresholds.

## Authorized Diagnostics

The first diagnostic layer reports:

1. year-by-year row-level:
   - known-target rows;
   - target prevalence;
   - mean predicted probability;
   - PR-AUC;
   - ROC-AUC;
   - Brier score;
2. year-by-year alert-episode accounting:
   - total alert episodes;
   - false alarms;
   - early detections;
   - late detections;
3. calibration/reliability table at a fixed, pre-declared 10-bin probability
   partition;
4. early-detection lead-time distribution summary.

These are descriptive analyses of the already-produced Final Test outputs.

## Forbidden Actions

Phase 9 must not:

- calculate an alternative "optimal" threshold;
- sweep thresholds;
- rank alternative models;
- generate new model probabilities;
- refit on Final Test labels;
- replace the official Phase 8 metrics;
- call a new operating point "Final Test performance";
- use post-hoc diagnostics as though they were pre-registered confirmatory
  results.

## Interpretation Rule

Any statement based on Phase 9 must be labeled post-hoc/exploratory.

If Phase 9 suggests a useful modification, that modification belongs to a new
future protocol and requires a new unseen evaluation period.
