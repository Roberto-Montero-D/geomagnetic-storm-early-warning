# Phase 9 Closure — Post-Hoc Scientific Diagnostics

**Project:** Geomagnetic Storm Early Warning System  
**Phase:** 9 — Post-Hoc Scientific Diagnostics  
**Status:** CLOSED  
**Analysis type:** POST-HOC / EXPLORATORY  
**Confirmatory result:** Phase 8 remains frozen and unchanged

---

## 1. Closure declaration

Phase 9 is formally closed.

No additional Phase 9 diagnostic is authorized under the current experiment.
The protected 2022–2025 Final Test has been consumed not only for the
pre-registered Phase 8 evaluation but also for post-hoc scientific
interpretation in Phase 9. It must therefore not be used to validate,
select, tune, or confirm any future modification suggested by these
diagnostics.

All Phase 9 findings are exploratory. They explain the observed protected-test
behavior; they do not replace or modify the Phase 8 result.

---

## 2. Governance boundary

The experiment preserves the following hierarchy:

1. **Phase 8** is the one-time confirmatory protected Final Test.
2. **Phase 9** is post-hoc interpretation of that result.
3. Any modification motivated by Phase 9 belongs to a **new future protocol**.
4. Such a modification requires evaluation on data not used to generate the
   Phase 9 hypotheses.

Phase 9 did not authorize:

- model selection;
- model retraining;
- threshold optimization;
- threshold sweeping;
- feature selection;
- replacement of the official Phase 8 operating point;
- reinterpretation of an alternative operating point as Final Test
  performance.

No Phase 9 diagnostic performed those actions.

---

## 3. Diagnostic sequence completed

Phase 9 ultimately comprised the following diagnostic layers.

### 3.1 Baseline post-hoc diagnostics

The initial diagnostic layer described the immutable Phase 8 outputs using:

- year-by-year probability metrics;
- year-by-year alert-episode accounting;
- fixed-bin calibration/reliability summaries;
- lead-time summaries.

This established the temporal structure of the protected-test performance
without changing the frozen operating point.

### 3.2 Physical error-regime diagnostics

Detected storms, missed storms, and false-alarm episodes were examined using
causal physical states from the frozen predictor universe.

The purpose was descriptive interpretation, not feature ranking.

### 3.3 Operational and temporal decomposition

Event outcomes were decomposed across protected-test years and operational
contexts. The analysis identified substantial temporal degradation,
particularly in 2025.

### 3.4 Event-context and recurrence diagnostics

Canonical events were stratified by recent storm recurrence and preceding
geomagnetic activity.

The strongest observed separation was associated with the geomagnetic
background immediately preceding the event.

### 3.5 Conditional physical-state diagnostics

Pre-onset physical states were compared between:

- detected and missed events;
- 2025 and 2022–2024;
- matched recurrence/background strata.

This tested whether the 2025 degradation could be explained solely by a
different mixture of event contexts.

### 3.6 Onset-centered signal-timing diagnostics

Frozen causal physical states and immutable Phase 8 probabilities were aligned
from -12 h through -1 h relative to canonical storm onset.

This tested whether missed storms merely revealed useful physical information
later than detected storms.

The simple late-information hypothesis was not supported.

---

## 4. Principal observed findings

### 4.1 Overall protected-event detection

Across the protected 2022–2025 event set used in the Phase 9 event-level
diagnostics:

- canonical events: **151**
- detected events: **82**
- missed events: **69**
- event recall: **54.3%**

These post-hoc counts are consistent with the frozen Phase 8 event outcomes and
do not constitute a new operating point.

### 4.2 Strong dependence on pre-event geomagnetic context

Event recall differed substantially according to recent activity:

| Context | Events | Detected | Event recall |
|---|---:|---:|---:|
| Isolated within 72 h | 102 | 66 | 64.7% |
| Recurrent within 72 h | 49 | 16 | 32.7% |
| No active Kp in prior 12 h | 70 | 54 | 77.1% |
| Active Kp in prior 12 h | 81 | 28 | 34.6% |

The 12-hour geomagnetic background was therefore a particularly strong
descriptive separator.

The joint recurrence/background analysis was more informative than recurrence
alone:

| Prior event within 72 h | Active Kp in prior 12 h | Events | Event recall |
|---|---|---:|---:|
| No | No | 62 | 75.8% |
| No | Yes | 40 | 47.5% |
| Yes | No | 8 | 87.5% |
| Yes | Yes | 41 | 22.0% |

The recurrent-plus-active regime was the poorest-performing substantial
subgroup.

The recurrent-but-quiet subgroup contained only eight events and must not be
overinterpreted.

### 4.3 Recall decreased with increasing background Kp

A strong descriptive gradient was observed as the median Kp background over
the preceding 12 hours increased:

| Median preceding Kp | Events | Event recall |
|---|---:|---:|
| <= 1.5 | 26 | 92.3% |
| 1.5–2.5 | 38 | 76.3% |
| 2.5–3.5 | 42 | 50.0% |
| 3.5–4.5 | 32 | 21.9% |
| > 4.5 | 13 | 7.7% |

This pattern is post-hoc and descriptive. It was not used to create a new
decision rule.

### 4.4 2025 degradation was not explained solely by event composition

Overall event recall changed from:

- **63.4%** in 2022–2024;
- to **36.0%** in 2025.

Conditional comparisons showed degradation even within isolated events:

| Context | 2022–2024 recall | 2025 recall |
|---|---:|---:|
| Isolated + quiet | 81.3% | 57.1% |
| Isolated + active | 60.9% | 29.4% |
| Recurrent + active | 17.4% | 27.8% |

The recurrent + quiet comparison is not interpretable because 2025 contained
only one event in that stratum.

The recurrent + active regime was already weak before 2025. Its recall did not
collapse further in 2025. Therefore, the additional 2025 degradation occurred
substantially outside this already-difficult structural regime.

### 4.5 2025 events occupied a shifted physical regime

Descriptive pre-onset comparisons showed that 2025 events generally occurred
under more energetic conditions than 2022–2024 events.

Across all events, representative median shifts included:

- last pre-onset solar-wind speed: approximately **433 -> 505 km/s**;
- median pre-onset solar-wind speed: approximately **418.5 -> 481.5 km/s**;
- median Bt: approximately **8.65 -> 10.55 nT**;
- median causal Kp: approximately **2.7 -> 3.3**.

Within recurrent + active events, the median pre-onset speed increased from
approximately **482 to 598.5 km/s**.

These results support a physical/distribution shift interpretation but do not
identify a causal mechanism by themselves.

### 4.6 Signal timing differed between detected and missed events

Phase 9.5 compared the early pre-onset interval (-12..-7 h) with the primary
warning interval (-6..-1 h).

For detected events, median within-event changes were:

- Bt: **+0.50 nT**
- Bz: **-0.525 nT**
- Bz-negative x speed: **+28.125**
- flow pressure: **+0.2675**
- causal Kp: **0.0**
- Phase 8 probability: **+0.00344**
- speed: **-0.5 km/s**

For missed events:

- Bt: **-0.10 nT**
- Bz: **-0.15 nT**
- Bz-negative x speed: **0**
- flow pressure: **0**
- causal Kp: **+0.3**
- Phase 8 probability: **+0.00830**
- speed: **+7 km/s**

Detected events therefore showed a clearer median development toward stronger
southward-field/coupling conditions during the warning window.

Missed events did not simply show the same signal shifted later.

### 4.7 Missed events were often already in disturbed states

The onset-centered trajectories showed that missed events frequently had
higher solar-wind speed, causal Kp, and model probability many hours before
canonical onset.

Representative median states at -12 h were approximately:

| Quantity | Detected | Missed |
|---|---:|---:|
| Solar-wind speed | 395 km/s | 479 km/s |
| Causal Kp | 2.0 | 3.3 |
| Phase 8 probability | 0.013 | 0.219 |

At -6 h, representative medians were approximately:

| Quantity | Detected | Missed |
|---|---:|---:|
| Solar-wind speed | 390 km/s | 466 km/s |
| Causal Kp | 2.0 | 3.7 |
| Phase 8 probability | 0.016 | 0.261 |

Thus many missed canonical onsets did not occur because the system saw an
entirely quiet environment. They often occurred while the environment and the
model were already indicating disturbed conditions.

### 4.8 Missed 2025 events showed additional temporal difficulty

For missed 2025 events, median changes from -12..-7 h to -6..-1 h included:

- Bz: **-0.175 nT**
- Bz-negative x speed: **0**
- Bt: **-0.50 nT**
- flow pressure: **-0.2375**
- causal Kp: **+0.25**
- Phase 8 probability: **-0.00998**
- speed: **+4.5 km/s**

The median model probability therefore did not simply rise too late; it
slightly decreased toward onset in this subgroup.

This supports a genuine temporal/generalization difficulty in addition to the
event-context problem.

---

## 5. Scientific interpretation

The Phase 9 evidence supports the following interpretation.

### 5.1 Structural event-context limitation

The frozen system performs best when a storm onset develops from a relatively
quiet background.

It performs poorly when a new canonical event begins while geomagnetic
conditions remain active following recent disturbance.

A plausible interpretation is that the predictor state naturally contains
information about whether the geospace environment is disturbed, while the
operational target requires the system to identify whether a **new canonical
storm onset** will occur.

Those are related but not identical questions.

This is especially relevant to recurrent + active events, for which the
physical state may resemble continuation or re-intensification rather than a
clean transition from quiet to storm conditions.

### 5.2 Temporal/distribution-shift limitation

The 2025 degradation cannot be attributed solely to a larger fraction of
recurrent or active-background events.

Performance also declined within isolated contexts, and the pre-onset physical
distribution shifted toward faster and generally more energized solar-wind
conditions.

Some missed 2025 storms also lacked the warning-window strengthening pattern
observed among successful detections.

The protected result therefore demonstrates meaningful temporal robustness
limitations.

### 5.3 Information timing alone is insufficient as an explanation

Phase 9.5 does not support the simple hypothesis that missed events become
predictable only during the final one or two hours before onset.

Instead, many misses occur in environments that are already disturbed well
before onset but do not exhibit a clean new transition during the warning
window.

---

## 6. Conclusions that are supported

The following conclusions are supported as **post-hoc exploratory findings**:

1. Protected-test performance is temporally heterogeneous.
2. 2025 is substantially harder than 2022–2024.
3. Elevated recent geomagnetic activity is strongly associated with lower
   event recall.
4. Recurrent + active events are a persistent structural weakness.
5. The 2025 degradation is not explained solely by a higher prevalence of that
   regime.
6. 2025 events show meaningful physical-state shifts relative to earlier
   protected-test years.
7. Detected events show a clearer median development of southward-field /
   coupling conditions during the primary warning window.
8. Missed events often occur in already-disturbed environments with elevated
   model probabilities well before canonical onset.
9. A simple "useful information arrives too late" explanation is insufficient.

---

## 7. Conclusions that are NOT supported

Phase 9 does **not** establish that:

- the canonical event definition is incorrect;
- the alert cooldown is incorrect;
- a different threshold would improve the scientific system;
- a specific new feature will improve future generalization;
- a specific alternative model would outperform the frozen model;
- recurrence-aware modeling will necessarily improve performance;
- background-relative features will necessarily improve performance;
- a change-point formulation is superior;
- 2025 represents a permanent future regime;
- any Phase 9 subgroup boundary should become an operational rule.

Those are hypotheses generated after inspecting the protected test.

They require a new protocol and new unseen evaluation data.

---

## 8. SHAP / feature-importance governance exception

The master protocol planned SHAP / feature-importance interpretation during
Phase 9.

This item was not executed.

The reason is methodological rather than computational:

1. the Phase 8 runner instantiated and fitted the frozen estimator in memory;
2. the fitted estimator was not persisted as an immutable Phase 8 artifact;
3. Phase 8 persisted predictions, alert episodes, and final metrics;
4. the Phase 9 diagnostics contract forbids model retraining.

Refitting an equivalent estimator after inspecting Final Test outcomes solely
to obtain SHAP values would violate the stronger Phase 9 no-refit governance
boundary.

Therefore:

> SHAP / feature-importance interpretation is recorded as a planned but
> unexecuted Phase 9 item because the fitted protected estimator was not
> persisted and post-hoc refitting is forbidden.

This omission does not invalidate the Phase 8 confirmatory metrics.

It is a reproducibility/interpretability limitation and must be corrected in a
future protocol by persisting the fitted protected estimator before protected
test scoring.

No retrospective refit is authorized for the current experiment.

---

## 9. Limitations exposed by Phase 9

### 9.1 Protected estimator was not persisted

This prevented protocol-planned SHAP analysis under the no-refit rule.

### 9.2 Subgroup analyses are post-hoc

The recurrence, background-Kp, year, and physical-regime analyses were
motivated by observed protected-test behavior. Their numerical differences are
descriptive and hypothesis-generating.

### 9.3 Some strata are small

In particular, recurrent + quiet events are uncommon. Percentages from very
small strata must not be interpreted as stable performance estimates.

### 9.4 Solar-cycle interpretation remains observational

Temporal differences are consistent with changing heliospheric/geomagnetic
regimes, but Phase 9 does not establish a causal solar-cycle mechanism.

### 9.5 Canonical-event semantics and physical disturbance are not identical

The analysis suggests a possible mismatch between discrete canonical event
boundaries and continuously disturbed physical states. Phase 9 does not resolve
which operational ontology would be preferable.

---

## 10. Quarantined hypotheses for a future protocol

The following ideas were generated from protected-test diagnostics and are
therefore **quarantined** from the current experiment.

They may be considered only in a newly pre-registered development cycle:

- background-relative or anomaly-from-recent-baseline features;
- explicit recent-event / recovery-state variables;
- change-point or onset-transition modeling;
- separate modeling of quiet-onset and disturbed-background regimes;
- recurrence-aware state representations;
- improved representation of recovery and re-intensification;
- explicit regime-robust training objectives;
- alternative event ontologies, only if scientifically justified before new
  evaluation;
- model persistence plus predeclared SHAP analysis;
- evaluation across solar-cycle / activity regimes;
- additional upstream solar/CME information if its real-time availability can
  be reconstructed causally.

These are **not validated improvements**.

No ranking among them is claimed.

---

## 11. Data-use declaration

The 2022–2025 period is now considered fully consumed for this project version.

It was used for:

1. the one-time Phase 8 protected Final Test;
2. Phase 9 post-hoc temporal diagnostics;
3. physical error-regime analysis;
4. recurrence/background analysis;
5. conditional 2025 analysis;
6. onset-centered signal-timing analysis.

Consequently:

> No future model, feature set, threshold, event definition, alert rule, or
> modeling strategy motivated by Phase 9 may claim independent confirmation
> using 2022–2025.

Future work must clearly distinguish:

- **development data** used to formulate and tune a new system;
- **new unseen confirmatory data** used to evaluate it.

If sufficiently new future observations are not yet available, future work may
perform development experiments, but those results must not be represented as
a new independent Final Test.

---

## 12. Required change for the next protocol

Before any future protected evaluation, the protocol should require immutable
persistence of at least:

- exact training-data identifiers / temporal bounds;
- feature-column order;
- fitted preprocessing objects, if any;
- fitted estimator;
- model configuration;
- software/environment metadata;
- operating threshold;
- generated protected probabilities;
- alert episodes;
- final metrics.

This allows predeclared interpretability analyses to be performed without
retraining after protected outcomes have been inspected.

---

## 13. Final Phase 9 statement

Phase 9 successfully transformed the protected-test generalization gap into a
set of scientifically interpretable, explicitly post-hoc findings.

The dominant evidence indicates that the system's failures are not explained
by one factor alone.

Two broad limitations emerged:

1. **event-context limitation** — identifying new canonical storm onsets during
   already disturbed or recurrent geomagnetic conditions;
2. **temporal/generalization limitation** — especially in 2025, where event
   composition and pre-onset physical states shifted and conditional
   performance deteriorated even outside the recurrent-active regime.

These findings do not alter the frozen Phase 8 result.

They define hypotheses for future work.

**Phase 9 is CLOSED.**
