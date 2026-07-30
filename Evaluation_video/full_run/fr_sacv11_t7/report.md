# Full Run Report

## Summary

- **Status:** Goal reached
- **Total time elapsed:** 54.0 s
- **Steps:** 540
- **Path length:** 50.0 m
- **Policy:** hybrid
- **Seed:** 19662
- **Encounters:** 3
- **Collisions:** 0
- **Avoided:** 3

- **Video:** `fr_sacv11_t7.gif`

## Encounter Results

| Encounter | Result | Min distance (m) | Path-follow min (m) |
|-----------|--------|------------------|---------------------|
| 1 | Success | 1.96 | 0.64 |
| 2 | Success | 1.61 | 0.48 |
| 3 | Success | 1.23 | 0.50 |

*Path-follow min* is the closest human–robot distance if the robot had not replanned (distance to the ghost that continues on the reference path).

## Stop-and-Wait Baseline

- **Stop-and-wait time-to-goal:** 69.5 s (unobstructed 48.9 s + 20.6 s waiting)
- **This policy's time-to-goal:** 54.0 s
- **Navigation efficiency improvement:** +22.3%

The purple ghost follows the reference path and freezes whenever a pedestrian comes within `human_detect_radius` -- the same trigger that hands control to the RL policy, so the two differ only in how they respond -- and resumes once the pedestrian has left its corridor. It is re-placed at the robot's on-path position at the onset of every encounter, so it meets each pedestrian with the robot's timing instead of arriving after the crossing is over. A zero velocity command freezes it in place and the controller is memoryless, so a stop-and-wait traversal is the unobstructed traversal with pauses inserted: its time-to-goal is the unobstructed time plus the time spent waiting.
