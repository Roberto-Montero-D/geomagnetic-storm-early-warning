# Phase 2.3 Protocol Gap — B1 Numerical Thresholds

`MASTER_PROTOCOL_v1.3.md` freezes B1 structurally as:

    Bz < -X AND V > Y

but does not specify numerical values for `X` or `Y`, nor does it specify a
candidate grid or a selection procedure for these parameters.

Therefore Phase 2.3 implementation intentionally:

- implements the rule exactly;
- uses only canonical causal `bz_gsm` and `speed`;
- requires X and Y explicitly;
- does not invent default values;
- does not search a grid;
- does not optimize using validation results;
- does not touch Final Test outcomes.

Before B1 can receive an official empirical baseline result, the project must
resolve this pre-existing protocol gap without using performance results to
choose the rule. That resolution should be recorded as a protocol clarification
or amendment before B1 evaluation.
