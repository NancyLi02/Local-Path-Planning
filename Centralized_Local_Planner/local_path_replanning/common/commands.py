"""Local command produced by the local replanner for one AMR.

The planner never drives the robot directly: it emits a command that a
tracking controller (simulator or real fleet manager) executes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MODE_TRACK = "TRACK"
MODE_STOP = "STOP"


@dataclass
class Command:
    mode: str                       # TRACK | STOP
    waypoints: np.ndarray           # (W, 2) world-frame path to follow
    target_speed: float             # m/s over the next control interval
    speed_limit: float              # m/s cap over the command horizon
    amr_id: str = ""
    action: np.ndarray | None = None        # decoded (goal_fwd, goal_lat, speed_scale)
    trajectory: object | None = None        # full Trajectory (for the shield / viz)
    shield_modified: bool = False           # proposal was overridden
    safe: bool = True                       # a safe candidate existed
    cost: float = float("inf")
    cost_terms: dict = field(default_factory=dict)
    reason: str = ""
    min_worker_clearance: float = float("inf")
    min_amr_distance: float = float("inf")

    def as_dict(self) -> dict:
        return dict(mode=self.mode, waypoints=self.waypoints,
                    target_speed=self.target_speed, speed_limit=self.speed_limit)


def trajectory_to_command(amr, trajectory, result=None, config=None,
                          waypoint_dt: float = 0.5) -> Command:
    """Convert an accepted trajectory into a dispatchable command."""
    dt = float(trajectory.times[1] - trajectory.times[0]) if len(trajectory.times) > 1 else 0.1
    stride = max(int(round(waypoint_dt / max(dt, 1e-9))), 1)
    # Only the command portion is dispatched; the tail exists so the shield can
    # judge the manoeuvre over the full prediction horizon.
    last = int(trajectory.command_steps or (len(trajectory.times) - 1))
    idx = np.arange(0, last + 1, stride)
    if idx[-1] != last:
        idx = np.append(idx, last)

    control_steps = max(int(round(0.2 / max(dt, 1e-9))), 1)
    target_speed = float(trajectory.speeds[1:control_steps + 1].mean()
                         if len(trajectory.speeds) > 1 else 0.0)
    speed_limit = float(trajectory.speeds[:last + 1].max())
    # STOP is decided by where the command ENDS: a stop command still carries a
    # braking profile, so ``target_speed`` stays non-zero while the AMR slows.
    stopped = float(trajectory.speeds[last]) < 1e-3

    cmd = Command(
        mode=MODE_STOP if stopped else MODE_TRACK,
        waypoints=trajectory.positions[idx].copy(),
        target_speed=target_speed,
        speed_limit=speed_limit,
        amr_id=getattr(amr, "id", ""),
        trajectory=trajectory,
    )
    if result is not None:
        cmd.action = np.asarray(result.action, dtype=float)
        cmd.shield_modified = bool(result.modified)
        cmd.safe = bool(result.safe)
        cmd.cost = float(result.cost)
        cmd.cost_terms = dict(result.cost_terms)
        cmd.reason = result.reason
        cmd.min_worker_clearance = float(result.report.min_worker_clearance)
        cmd.min_amr_distance = float(result.report.min_amr_distance)
    return cmd
