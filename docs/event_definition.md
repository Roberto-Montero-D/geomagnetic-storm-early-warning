# Event Definition

## Storm Episodes

- Start: First hour where Kp >= T
- End: Last hour before Kp < T for Z consecutive hours
- Z = 6 hours

## Counting Rule

Each event is counted once, regardless of the number of alert episodes that overlap it.

## Function

`identify_events(kp_series, threshold=T, cooldown=Z)`