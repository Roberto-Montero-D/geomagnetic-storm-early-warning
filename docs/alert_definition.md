# Alert Definition

## Alert Episodes

- Hourly alert: P(storm) >= tau
- Alert episode: Group of consecutive alerts
- Cooldown (C): 3 hours between episodes

## Classification

| Condition | Classification |
|-----------|---------------|
| first_alert < storm_start - H | False Alarm |
| storm_start - H <= first_alert < storm_start | Early Detection |
| storm_start <= first_alert <= storm_end | Late Detection |
| No overlap | False Alarm |

## Function

`identify_alerts(prob_series, threshold=tau, cooldown=C, H=H)`