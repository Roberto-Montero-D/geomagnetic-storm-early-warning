# Phase 3 Initial Feature-Screening Results

**Protocol:** `MASTER_PROTOCOL_v1.3.md`  
**Contract:** `docs/phase3_feature_screening_contract.md`  
**Stage:** Initial screening, 1996–2016 -> 2017–2018  
**Status:** Complete and audited; advancing candidates frozen before walk-forward confirmation.

## 1. Frozen Experiment Definitions

| Experiment | Feature families | Number of features |
|---|---|---:|
| A | Raw | 10 |
| B | Raw + Rolling | 70 |
| C | Raw + Rolling + Persistence | 75 |
| D | Raw + Rolling + Persistence + Dynamics | 90 |
| E | Raw + Rolling + Persistence + Dynamics + Interactions | 93 |

All experiments used the same frozen unbalanced ExtraTrees configuration:

```text
n_estimators = 100
max_depth = 10
class_weight = None
random_state = 42
```

## 2. Official Initial-Screening Results

| Rank | Experiment | Threshold | Event Recall | FAR/day | PR-AUC |
|---:|---|---:|---:|---:|---:|
| 1 | **A** | 0.09 | **0.522727** | 0.191159 | 0.468504 |
| 2 | **E** | 0.08 | 0.477273 | 0.195285 | **0.479092** |
| 3 | **C** | 0.07 | 0.431818 | 0.184283 | 0.474958 |
| 4 | B | 0.07 | 0.431818 | **0.182908** | 0.469869 |
| 5 | D | 0.07 | 0.409091 | 0.192535 | 0.473434 |

All five experiments were operationally feasible under the frozen
`FAR/day <= 0.2` screening constraint.

## 3. Threshold-Boundary Audit

| Experiment | Previous tau | Previous FAR/day | Selected tau | Selected FAR/day |
|---|---:|---:|---:|---:|
| A | 0.08 | 0.211787 | **0.09** | **0.191159** |
| B | 0.06 | 0.206286 | **0.07** | **0.182908** |
| C | 0.06 | 0.200785 | **0.07** | **0.184283** |
| D | 0.06 | 0.200785 | **0.07** | **0.192535** |
| E | 0.07 | 0.203535 | **0.08** | **0.195285** |

For every experiment, the immediately preceding grid threshold violates the
FAR constraint and the selected threshold satisfies it. Therefore each
selected threshold is the minimum feasible threshold on the frozen
`0.01..0.99` grid.

## 4. Frozen Advancement Decision

The Phase 3 contract requires exactly three candidates to advance when at
least three experiments are feasible. Ranking priority was frozen before
performance inspection as:

1. higher Event Recall;
2. higher PR-AUC;
3. lower FAR/day;
4. smaller feature set.

The advancing candidates are therefore, in frozen screening rank order:

```text
1. A — Raw
2. E — Raw + Rolling + Persistence + Dynamics + Interactions
3. C — Raw + Rolling + Persistence
```

The C/B tie on Event Recall is resolved by the second frozen criterion:
C has higher PR-AUC than B.

**A, E, and C are now frozen as the only candidates entering Phase 3
walk-forward confirmation.** This candidate set must not be changed in
response to later confirmation results.

## 5. Interpretation Boundary

The initial screening does not establish that A is the final best feature
set. Its first-place result applies only to the frozen 2017–2018 screening
period.

The purpose of the next stage is temporal confirmation on:

```text
1996–2018 -> 2019–2020
1996–2020 -> 2021
```

No explanation for differences among A–E is inferred from this screening
stage alone.

## 6. Reproducibility Artifacts

The official screening run generated:

```text
results/phase3/screening/screening_ranking.csv
results/phase3/screening/screening_advancing_experiments.csv
results/phase3/screening/screening_metrics.csv
results/phase3/screening/screening_a_threshold_curve.csv
results/phase3/screening/screening_b_threshold_curve.csv
results/phase3/screening/screening_c_threshold_curve.csv
results/phase3/screening/screening_d_threshold_curve.csv
results/phase3/screening/screening_e_threshold_curve.csv
```

These are development-only screening artifacts and do not export target
timelines, prediction timestamps, or protected Final Test outcomes.

## 7. Freeze Boundary

From this point onward, the following initial-screening decisions are frozen:

- advancing candidates: **A, E, C**;
- initial screening metrics and thresholds;
- A–E feature manifests;
- screening model configuration;
- screening ranking/tie rules;
- FAR constraint.

The next methodological step is implementation of walk-forward confirmation.
