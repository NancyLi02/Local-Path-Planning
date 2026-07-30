# Full Run Report

## Summary

- **Status:** Goal reached
- **Total time elapsed:** 88.8 s
- **Steps:** 888
- **Path length:** 80.0 m
- **Policy:** hybrid
- **Seed:** 42
- **Encounters:** 6
- **Collisions:** 1
- **Avoided:** 5

- **Video:** `fr_sacv11_t4.gif`

## Encounter Results

| Encounter | Result | Min distance (m) | Path-follow min (m) |
|-----------|--------|------------------|---------------------|
| 1 | Success | 1.00 | 0.33 |
| 2 | Success | 1.58 | 0.62 |
| 3 | Success | 1.01 | 0.18 |
| 4 | Success | 0.99 | 0.25 |
| 5 | Success | 1.15 | 0.61 |
| 6 | Collision | 0.68 | 0.19 |

*Path-follow min* is the closest human–robot distance if the robot had not replanned (distance to the ghost that continues on the reference path).

## Stop-and-Wait Baseline

- **Stop-and-wait time-to-goal:** 126.7 s (unobstructed 78.9 s + 47.8 s waiting)
- **This policy's time-to-goal:** 88.8 s
- **Navigation efficiency improvement:** +29.9%

The purple ghost follows the reference path and freezes whenever a pedestrian comes within `human_detect_radius` -- the same trigger that hands control to the RL policy, so the two differ only in how they respond -- and resumes once the pedestrian has left its corridor. It is re-placed at the robot's on-path position at the onset of every encounter, so it meets each pedestrian with the robot's timing instead of arriving after the crossing is over. A zero velocity command freezes it in place and the controller is memoryless, so a stop-and-wait traversal is the unobstructed traversal with pauses inserted: its time-to-goal is the unobstructed time plus the time spent waiting.
