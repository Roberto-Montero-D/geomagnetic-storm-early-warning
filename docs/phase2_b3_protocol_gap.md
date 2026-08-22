# Phase 2.5 Protocol Gap — B3 ExtraTrees Baseline Configuration

**Status:** RESOLVED

`MASTER_PROTOCOL_v1.3.md` froze B3 structurally as:

```text
ExtraTrees
raw primary features
no balancing
```

but did not freeze one numerical baseline ExtraTrees configuration.

The later Phase 5 model-selection grid specifies:

```text
n_estimators = 100, 200, 500
max_depth = 10, 20, None
```

That later search grid was not treated as though a winning Phase 2 baseline
configuration had already been selected.

The gap was resolved **before official empirical Phase 2 baseline evaluation**
in:

```text
docs/phase2_baseline_configuration_freeze.md
```

The frozen primary B3 configuration is now:

```text
n_estimators = 100
max_depth = 10
class_weight = None
random_state = 42
```

The original gap record is retained for provenance. It no longer represents an
open methodological issue.
