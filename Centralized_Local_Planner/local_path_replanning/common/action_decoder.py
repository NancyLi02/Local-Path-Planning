"""Bounded decoding of the raw policy output into a local replanning action.

    raw a_i  ->  a_i = (goal_fwd, goal_lat, speed_scale)

    goal_fwd    = max_forward_dist   * sigmoid(raw[..., 0])   in [0, F]
    goal_lat    = max_lateral_offset * tanh(raw[..., 1])      in [-L, L]
    speed_scale = sigmoid(raw[..., 2])                        in [0, 1]

The same decoding is used by the learned network (torch) and by the rule planners /
shield backups (numpy), so learned and hand-built actions are interchangeable.
"""
from __future__ import annotations

import math

import numpy as np

try:                                     # torch is optional for the rule planners
    import torch
except Exception:                        # pragma: no cover
    torch = None


FWD, LAT, SPD = 0, 1, 2


def decode_action(raw_action, max_forward_dist: float, max_lateral_offset: float):
    """Torch version (spec interface). raw_action: [..., 3] -> [..., 3]."""
    goal_fwd = max_forward_dist * torch.sigmoid(raw_action[..., 0:1])
    goal_lat = max_lateral_offset * torch.tanh(raw_action[..., 1:2])
    speed_scale = torch.sigmoid(raw_action[..., 2:3])
    return torch.cat([goal_fwd, goal_lat, speed_scale], dim=-1)


def decode_action_np(raw_action: np.ndarray, max_forward_dist: float,
                     max_lateral_offset: float) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=float)
    fwd = max_forward_dist * _sigmoid(raw[..., 0])
    lat = max_lateral_offset * np.tanh(raw[..., 1])
    spd = _sigmoid(raw[..., 2])
    return np.stack([fwd, lat, spd], axis=-1)


def encode_action_np(action: np.ndarray, max_forward_dist: float,
                     max_lateral_offset: float) -> np.ndarray:
    """Inverse of :func:`decode_action_np` (used for behaviour cloning targets)."""
    a = np.asarray(action, dtype=float)
    fwd = np.clip(a[..., 0] / max(max_forward_dist, 1e-9), 1e-4, 1 - 1e-4)
    lat = np.clip(a[..., 1] / max(max_lateral_offset, 1e-9), -1 + 1e-4, 1 - 1e-4)
    spd = np.clip(a[..., 2], 1e-4, 1 - 1e-4)
    return np.stack([_logit(fwd), np.arctanh(lat), _logit(spd)], axis=-1)


def action_to_local_goal(agent, action, config) -> tuple[np.ndarray, float, float]:
    """Convert a decoded action into a world-frame local goal.

        target = path_point(s + goal_fwd) + goal_lat * path_normal

    Returns ``(target_xy, goal_s, goal_lat)``.
    """
    goal_fwd = float(action[FWD])
    goal_lat = float(np.clip(action[LAT], -config.max_lateral_offset,
                             config.max_lateral_offset))
    goal_s = float(min(agent.s + goal_fwd, agent.total_length))
    return agent.path_position(goal_s, goal_lat), goal_s, goal_lat


def clip_action(action: np.ndarray, config) -> np.ndarray:
    a = np.asarray(action, dtype=float).copy()
    a[..., FWD] = np.clip(a[..., FWD], 0.0, config.max_forward_dist)
    a[..., LAT] = np.clip(a[..., LAT], -config.max_lateral_offset,
                          config.max_lateral_offset)
    # Policy actions decode to [0, 1]; only the shield's emergency reverse
    # candidate ever reaches the negative part of the range.
    lo = -(config.reverse_speed / max(config.v_max, 1e-9)) \
        if getattr(config, "allow_emergency_reverse", False) else 0.0
    a[..., SPD] = np.clip(a[..., SPD], lo, 1.0)
    return a


def reverse_action(agent, config, lat: float | None = None) -> np.ndarray:
    """Emergency back-off along the reference path.

    ``lat`` selects the lateral offset to hold while reversing; the shield
    offers "straight back", "back and left" and "back and right", because
    retreating along the lane alone does not help when the worker is walking
    in the same direction the AMR is retreating.
    """
    return np.array([config.max_forward_dist,
                     float(agent.lat) if lat is None else float(lat),
                     -config.reverse_speed / max(config.v_max, 1e-9)])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _logit(p):
    return np.log(p / (1.0 - p))
