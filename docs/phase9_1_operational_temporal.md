# Phase 9.1 — Operational Temporal Decomposition

**Status:** POST-HOC / EXPLORATORY  
**Official Phase 8 result changed:** NO

Phase 9.1 decomposes the frozen Phase 8 operational result by calendar year.

It does not train a model, regenerate probabilities, sweep thresholds, or
select a new operating point.

## Outputs

`yearly_operational_decomposition.csv` reports, for each year from 2022
through 2025:

- exact calendar exposure days;
- canonical storm-event count;
- unique detected and missed events;
- Event Recall;
- events with early and/or late detection;
- total alert episodes;
- false-alarm episodes;
- FAR/day;
- early- and late-detection episode counts;
- median early lead time.

`event_outcomes.csv` contains one row for each canonical protected storm event
and records whether it was detected, whether it received an early or late
detection, the number of associated detection episodes, the first detection
time, and maximum early lead.

## Temporal attribution

Storm events are attributed to the calendar year containing canonical
`start_time`.

False alarms are attributed to the calendar year containing episode
`first_alert_time`.

Annual FAR/day is:

```text
false-alarm episodes in calendar year / exact calendar exposure days
```

including the 366-day exposure for leap year 2024.

## Scientific restriction

These results are explanatory diagnostics of the already-consumed Final Test.
They cannot replace the Phase 8 metrics or be used to select a new threshold.
