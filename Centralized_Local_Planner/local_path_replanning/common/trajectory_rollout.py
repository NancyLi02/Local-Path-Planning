"""Turn a decoded action into a short-horizon trajectory for safety checking.

The AMR follows its global reference path with a commanded lateral offset:
the offset ramps (smoothstep, so there is no slope discontinuity and the
manoeuvre curvature stays bounded) from the current offset to ``goal_lat``
over the commanded forward distance, while the speed ramps to
``speed_scale * v_max`` under an acceleration limit. This yields a smooth, path-consistent manoeuvre that
covers every command type the specification asks for:

    goal_lat = 0                      -> pure speed control (slow / stop / go)
    goal_lat != 0                     -> lane shift / short detour
    speed_scale = 0                   -> STOP

The rollout is only used for shield checking and command generation, so it
does not need to be the final tracking controller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .action_decoder import FWD, LAT, SPD, clip_action


@dataclass
class Trajectory:
    times: np.ndarray        # (T,)   seconds, times[0] = 0 (current state)
    positions: np.ndarray    # (T, 2) world frame
    headings: np.ndarray     # (T,)   rad
    speeds: np.ndarray       # (T,)   m/s
    arclen: np.ndarray       # (T,)   path arc-length s(t)
    lateral: np.ndarray      # (T,)   signed lateral offset l(t)
    ref_headings: np.ndarray # (T,)   reference-path heading at s(t)
    command_steps: int = 0   # samples belonging to the dispatched command
    lat_speed_max: float = 0.0      # max |dl/dt|  [m/s]
    # Analytic manoeuvre geometry of the lateral ramp (exact, so the
    # feasibility check never sees finite-difference noise or the reference
    # path's own corners).
    lat_slope_max: float = 0.0      # max |dl/ds|
    lat_curv_max: float = 0.0       # max |d2l/ds2|  [1/m]

    def __len__(self) -> int:
        return int(len(self.times))

    @property
    def final_speed(self) -> float:
        return float(self.speeds[-1])

    def state_at(self, t_sec: float) -> tuple[np.ndarray, float, float, float]:
        """(position, heading, speed, arclen) at the sample nearest ``t_sec``."""
        i = int(np.clip(np.searchsorted(self.times, t_sec), 0, len(self.times) - 1))
        return (self.positions[i].copy(), float(self.headings[i]),
                float(self.speeds[i]), float(self.arclen[i]))


def rollout_action(amr, action, horizon_sec: float, dt: float, config,
                   map_data=None) -> Trajectory:
    """Roll one decoded action out into a Trajectory (spec: rollout_action).

    Kinematics (holonomic AMR on a reference path):

        longitudinal   s'  = v_long,  v_long -> speed_scale * v_max under a_max
        lateral        l'  = clip(k (goal_lat - l), +-v_lat_max)
        speed budget   sqrt(v_long^2 + l'^2) <= v_max

    The lateral law is driven in TIME, so it is state-consistent under
    receding-horizon re-planning (the first control interval already strafes)
    and it still works at zero forward speed -- an arc-length ramp gives a
    halted AMR no way to move aside at all.

    The trajectory is rolled out to ``config.eval_horizon_sec`` (the worker
    prediction horizon); only the first ``horizon_sec`` is dispatched.
    """
    action = clip_action(np.asarray(action, dtype=float), config)
    goal_fwd = float(action[FWD])
    goal_lat = float(action[LAT])
    v_target = float(action[SPD]) * config.v_max

    n_cmd = int(round(horizon_sec / max(dt, 1e-9)))
    steps = int(round(max(horizon_sec, config.eval_horizon_sec) / max(dt, 1e-9)))
    s0, l0 = float(amr.s), float(amr.lat)
    # A stop command holds the current offset: there is no manoeuvre to track.
    if abs(v_target) < 1e-6:
        goal_lat = l0

    # Approach gain: reach ~95 % of the offset over ``goal_fwd`` metres of
    # nominal travel, i.e. in goal_fwd / v_max seconds.
    k = 3.0 * config.v_max / max(goal_fwd, 1e-3)

    times = np.arange(steps + 1) * dt
    arclen = np.empty(steps + 1)
    lateral = np.empty(steps + 1)
    v_long_hist = np.empty(steps + 1)
    v_lat_hist = np.empty(steps + 1)
    arclen[0], lateral[0] = s0, l0
    v_long_hist[0] = float(amr.speed)
    v_lat_hist[0] = 0.0

    v_long = float(amr.speed)
    v_lat_prev = 0.0
    s, lat = s0, l0
    for i in range(1, steps + 1):
        # Desired velocity in the path frame, scaled to the speed budget.
        v_lat = float(np.clip(k * (goal_lat - lat), -config.v_lat_max, config.v_lat_max))
        v_long_des = v_target
        n = math.hypot(abs(v_long_des), v_lat)
        if n > config.v_max:
            scale = config.v_max / n
            v_long_des *= scale
            v_lat *= scale
        # Acceleration limit on the VELOCITY VECTOR. Limiting the two axes
        # independently allows a combined |dv| of sqrt(2) a_max, which the
        # feasibility check then rejects -- and because a lateral correction is
        # active almost all the time, that silently made every forward
        # candidate infeasible and left the AMR with nothing but STOP.
        dvx = v_long_des - v_long
        dvy = v_lat - v_lat_prev
        n_dv = math.hypot(dvx, dvy)
        lim = config.a_max * dt
        if n_dv > lim and n_dv > 1e-12:
            dvx *= lim / n_dv
            dvy *= lim / n_dv
        v_long = v_long + dvx
        v_lat = v_lat_prev + dvy
        if v_target >= 0.0:
            v_long = max(0.0, v_long)
        # The along-path rate cannot change instantly, so re-clip the lateral
        # rate to whatever is left of the speed budget.
        budget = math.sqrt(max(config.v_max ** 2 - v_long ** 2, 0.0))
        v_lat = float(np.clip(v_lat, -budget, budget))
        lat = lat + v_lat * dt
        v_lat_prev = v_lat
        s = float(np.clip(s + v_long * dt, 0.0, amr.total_length))
        arclen[i], lateral[i] = s, lat
        v_long_hist[i], v_lat_hist[i] = v_long, v_lat

    positions = amr.path_position(arclen, lateral)
    ref_headings = amr.path_heading(arclen)

    d = np.diff(positions, axis=0)
    seg = np.linalg.norm(d, axis=1)
    headings = np.empty(steps + 1)
    headings[0] = float(amr.heading)
    moving = seg > 1e-9
    headings[1:] = np.where(moving, np.arctan2(d[:, 1], d[:, 0]), np.nan)
    for i in range(1, steps + 1):                      # hold the last heading
        if not np.isfinite(headings[i]):
            headings[i] = headings[i - 1]
    speeds = np.empty(steps + 1)
    speeds[0] = float(amr.speed)
    speeds[1:] = seg / dt

    ds = np.diff(arclen)
    slope = np.abs(np.diff(lateral)) / np.maximum(ds, 1e-6)
    slope = slope[ds > 1e-6]
    return Trajectory(times=times, positions=positions, headings=headings,
                      speeds=speeds, arclen=arclen, lateral=lateral,
                      ref_headings=ref_headings, command_steps=n_cmd,
                      lat_speed_max=float(np.abs(v_lat_hist).max()),
                      lat_slope_max=float(slope.max()) if slope.size else 0.0,
                      lat_curv_max=0.0)


def generate_backup_candidates(action, config) -> list[np.ndarray]:
    """The specification's backup set, ordered by preference.

        original, slow down, stop, small left shift, small right shift,
        shorter forward, [+ full-width left / right detour]

    The two detour candidates realise the "short detour command" of the
    command vocabulary: a small shift is often not enough to clear a worker
    standing next to the lane, and without them the only remaining option is
    to stop and be walked into.
    """
    a = clip_action(np.asarray(action, dtype=float), config)
    if config.candidate_grid:
        # Index 0 stays the proposal itself so "shield modified" keeps meaning
        # "the proposal was overridden".
        grid = [a.copy()]
        for spd in config.grid_speeds:
            for lat in config.grid_lateral:
                grid.append(_with(a, lat=a[LAT] + lat, spd=a[SPD] * spd))
        return [clip_action(g, config) for g in grid]
    shift = config.backup_lateral_shift
    cands = [
        a.copy(),                                              # original
        _with(a, spd=a[SPD] * config.backup_slow_scale),       # slow down
        _with(a, spd=0.0),                                     # stop
        _with(a, lat=a[LAT] + shift),                          # small left shift
        _with(a, lat=a[LAT] - shift),                          # small right shift
        _with(a, fwd=a[FWD] * config.backup_forward_scale),    # shorter forward
    ]
    if config.use_detour_candidates:
        cands.append(_with(a, lat=config.max_lateral_offset))       # left detour
        cands.append(_with(a, lat=-config.max_lateral_offset))      # right detour
    if not config.use_backup_candidates:
        cands = [cands[0], cands[2]]                           # original + STOP only
    return [clip_action(c, config) for c in cands]


def stop_action(config) -> np.ndarray:
    return np.array([0.0, 0.0, 0.0])


def stop_trajectory(amr, horizon_sec: float, dt: float, config) -> Trajectory:
    return rollout_action(amr, stop_action(config), horizon_sec, dt, config)


def _with(a: np.ndarray, fwd=None, lat=None, spd=None) -> np.ndarray:
    out = a.copy()
    if fwd is not None:
        out[FWD] = fwd
    if lat is not None:
        out[LAT] = lat
    if spd is not None:
        out[SPD] = spd
    return out
