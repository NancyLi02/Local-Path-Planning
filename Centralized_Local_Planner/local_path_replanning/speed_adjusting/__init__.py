"""Speed adjusting: modulate speed along the rail; the AMR never leaves it.

    planner.py    the replanner itself (discrete speed factors + space-time shield)
    evaluate.py   its own evaluation: no-shield baseline, shield-parameter tuning
    adapter.py    runs it inside the shared comparison harness (same metric
                  vocabulary and the same renderer as the other three methods)
"""
from .planner import SpeedAdjustingConfig, SpeedAdjustingReplanner
__all__ = ["SpeedAdjustingReplanner", "SpeedAdjustingConfig"]
