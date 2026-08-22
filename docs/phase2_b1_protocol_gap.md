# Phase 2.3 Protocol Gap — B1 Numerical Thresholds

**Status:** RESOLVED

`MASTER_PROTOCOL_v1.3.md` froze B1 structurally as:

```text
Bz < -X AND V > Y
```

but did not specify numerical values for `X` or `Y`, nor a candidate grid or a
selection procedure for those parameters.

During Phase 2 implementation this was treated as a methodological gap rather
than silently choosing values from observed performance.

The gap was resolved **before official empirical Phase 2 baseline evaluation**
in:

```text
docs/phase2_baseline_configuration_freeze.md
```

The frozen primary B1 configuration is now:

```text
Bz < -5 nT
AND
V > 500 km/s
```

Both inequalities remain strict.

The original gap record is retained for provenance. It no longer represents an
open methodological issue.
