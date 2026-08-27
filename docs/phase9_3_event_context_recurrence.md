# Phase 9.3 — Event Context and Recurrence Diagnostics

**Status:** POST-HOC / EXPLORATORY  
**Official Phase 8 result changed:** NO

Phase 9.3 tests the descriptive hypothesis raised by Phase 9.2: the frozen
system may be less effective for storm onsets embedded in persistent or
recurrent geomagnetic activity than for isolated onsets.

## Predeclared context measures

For every protected Final Test event:

- hours since previous canonical event start;
- hours since previous canonical event end;
- number of prior canonical event starts within 24, 48, and 72 hours;
- median and maximum causal Kp background during the preceding 12 hours;
- number and fraction of those hours with causal Kp >= 4.

The current event start is excluded from every lookback.

## Predeclared strata

Two recurrence strata are reported:

- isolated: no prior canonical event start within 72 h;
- recurrent: at least one prior canonical event start within 72 h.

Two background-activity strata are also reported:

- no active Kp in the preceding 12 h;
- at least one hour with causal Kp >= 4 in the preceding 12 h.

Event Recall is reported descriptively for each stratum. These strata are not
used to alter predictions or select an operating point.

## Restrictions

No model fitting, threshold search, feature selection, inferential hypothesis
testing, or post-test optimization is authorized. Phase 9.3 can generate
hypotheses for a future protocol only; it cannot rehabilitate or replace the
failed Phase 8 FAR constraint.
