# Phase 2.5 Protocol Gap — B3 ExtraTrees Baseline Configuration

`MASTER_PROTOCOL_v1.3.md` freezes B3 structurally as:

    ExtraTrees, raw primary features, no balancing

but does not freeze one numerical baseline ExtraTrees configuration.

The later Phase 5 model-selection grid specifies:

    n_estimators = 100, 200, 500
    max_depth = 10, 20, None

That grid belongs to model selection and must not be silently reused as if one
of its combinations had already been selected for the Phase 2 baseline.

Therefore Phase 2.5 implementation intentionally:

- uses only the 10 canonical raw primary predictors;
- keeps `class_weight=None`;
- requires `n_estimators` and `max_depth` explicitly;
- does not choose a combination from observed validation performance;
- does not inspect Final Test outcomes;
- uses only a fixed random seed for reproducibility.

Before B3 receives one official empirical baseline result, the project should
record a protocol clarification/amendment that fixes a single B3 configuration
independently of observed model performance.
