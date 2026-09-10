"""Optimization-based: speed + lateral offset, chosen by cost over a horizon.

Same 3-DOF action space as ``learning_based`` -- the two differ only in who
proposes the action, which is what makes them directly comparable.

NOTE: ``planner.py`` currently holds a greedy stand-in (nominal proposal +
candidate-set cost search through the shield). It is the slot for a proper MPC
formulation; replacing it does not touch the runtime, the shield, the
evaluation harness or the renderer.
"""
from .planner import OptimizationBasedReplanner
__all__ = ["OptimizationBasedReplanner"]
