"""Per-AMR observation builder for the variable-N attention policy.

Feature blocks (specification layout):

    ego      6   [s_long, s_lat, theta_rel, v, a, omega]        path-relative
    goal     2   [ds_goal, dl_goal]                             path frame
    route   30   M=10 preview waypoints in the AMR local frame  (dx, dy, dtheta)
    worker  21   top-K=3 risky workers x 7 risk features
    priority 5   [ttc, 1/(ttc+eps), task_priority, braking_risk, affected]
    ---------------------------------------------------------------------
    spatial  4   [x, y, cos h, sin h]  (optional, config.include_spatial_block)

Every feature is normalised to roughly [-1, 1] before it reaches the network.
The optional spatial block is what lets the self-attention relate two AMRs
geometrically; disable it to reproduce the exact 64-dim specification vector.
"""
from __future__ import annotations

import math

import numpy as np

from Centralized_Local_Planner.tools.geometry import point_in_polygon

_EPS = 1e-9
_D_REF = 3.0            # m, distance at which a worker stops being "risky"


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def extract_ego_feature(amr, config) -> np.ndarray:
    return np.array([
        np.clip(amr.s / max(amr.total_length, _EPS), 0.0, 1.0),
        np.clip(amr.lat / max(config.max_lateral_offset, _EPS), -2.0, 2.0),
        amr.heading_rel / math.pi,
        np.clip(amr.speed / max(config.v_max, _EPS), -2.0, 2.0),
        np.clip(amr.accel / max(config.a_max, _EPS), -2.0, 2.0),
        np.clip(amr.omega / max(config.omega_max, _EPS), -2.0, 2.0),
    ], dtype=np.float32)


def extract_goal_feature(amr, config) -> np.ndarray:
    ds = float(amr.goal_s - amr.s)
    dl = float(0.0 - amr.lat)          # the local goal sits on the reference path
    return np.array([
        np.clip(ds / max(config.max_forward_dist, _EPS), -2.0, 2.0),
        np.clip(dl / max(config.max_lateral_offset, _EPS), -2.0, 2.0),
    ], dtype=np.float32)


def extract_route_feature(amr, map_data, config) -> np.ndarray:
    """Next M reference waypoints expressed in the AMR local frame."""
    M = config.route_preview_points
    spacing = config.max_forward_dist / max(M, 1)
    feat = np.zeros((M, 3), dtype=np.float32)
    h0 = amr.heading
    for m in range(M):
        s_m = min(amr.s + (m + 1) * spacing, amr.total_length)
        local = amr.to_local(amr.path_position(s_m, 0.0))
        dth = amr.ref.heading_at(s_m) - h0
        dth = (dth + math.pi) % (2 * math.pi) - math.pi
        feat[m] = (local[0] / config.max_forward_dist,
                   local[1] / config.max_forward_dist,
                   dth / math.pi)
    return feat.reshape(-1)


def extract_worker_tube_feature(amr, worker_predictions, config) -> np.ndarray:
    """Top-K risky workers, 7 features each.

        [d_min, t_min, dx_risk, dy_risk, sigma_x, sigma_y, risk_score]

    ``d_min`` / ``t_min`` are the closest approach between the AMR's
    constant-speed path rollout and the worker's predicted tube centres;
    ``dx/dy_risk`` is the relative worker position at that moment in the AMR
    local frame; ``sigma`` is the prediction uncertainty there.
    """
    K = config.max_workers
    per = config.worker_feature_dim_per_worker
    feat = np.zeros((K, per), dtype=np.float32)
    if not worker_predictions:
        return feat.reshape(-1)

    T = min(config.pred_horizon_steps,
            min(w.horizon for w in worker_predictions))
    ego = amr.rail_rollout(config.dt_pred, T)
    t_horizon = T * config.dt_pred

    rows = []
    for w in worker_predictions:
        d = np.linalg.norm(ego - w.centers[:T], axis=1)
        t_idx = int(np.argmin(d))
        d_min = float(d[t_idx])
        t_min = float((t_idx + 1) * config.dt_pred)
        rel = amr.to_local(w.centers[t_idx])
        sx, sy = w.sigma_at(t_idx)
        hard_hit = any(point_in_polygon(ego[t], w.hard_lobes[t]) for t in range(T))
        risk = (np.clip(1.0 - d_min / _D_REF, 0.0, 1.0)
                * np.clip(1.0 - t_min / max(t_horizon, _EPS), 0.0, 1.0))
        risk = float(np.clip(risk + (0.5 if hard_hit else 0.0), 0.0, 1.0))
        rows.append((risk, np.array([
            np.clip(d_min / 5.0, 0.0, 1.0),
            np.clip(t_min / max(t_horizon, _EPS), 0.0, 1.0),
            np.clip(rel[0] / 5.0, -1.0, 1.0),
            np.clip(rel[1] / 5.0, -1.0, 1.0),
            np.clip(sx / 2.0, 0.0, 1.0),
            np.clip(sy / 2.0, 0.0, 1.0),
            risk,
        ], dtype=np.float32)))

    rows.sort(key=lambda r: -r[0])
    for k in range(min(K, len(rows))):
        feat[k] = rows[k][1]
    return feat.reshape(-1)


def extract_priority_feature(amr, config) -> np.ndarray:
    t_max = config.pred_horizon_steps * config.dt_pred
    ttc = amr.ttc if np.isfinite(amr.ttc) else t_max
    return np.array([
        np.clip(ttc / max(t_max, _EPS), 0.0, 1.0),
        np.clip((1.0 / (ttc + config.ttc_eps)) * config.ttc_eps, 0.0, 1.0),
        np.clip(amr.task_priority, 0.0, 1.0),
        np.clip(amr.braking_risk, 0.0, 1.0),
        1.0 if amr.affected else 0.0,
    ], dtype=np.float32)


def extract_spatial_feature(amr, map_data, config) -> np.ndarray:
    xmin, xmax, ymin, ymax = map_data.bounds
    p = amr.position
    return np.array([
        (float(p[0]) - xmin) / max(xmax - xmin, _EPS),
        (float(p[1]) - ymin) / max(ymax - ymin, _EPS),
        math.cos(amr.heading), math.sin(amr.heading),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_amr_observation(amr, cluster, worker_predictions, map_data,
                          config) -> np.ndarray:
    """Concatenated observation vector for one AMR (shape ``[obs_dim]``)."""
    blocks = [
        extract_ego_feature(amr, config),
        extract_goal_feature(amr, config),
        extract_route_feature(amr, map_data, config),
        extract_worker_tube_feature(amr, worker_predictions, config),
        extract_priority_feature(amr, config),
    ]
    if config.include_spatial_block:
        blocks.append(extract_spatial_feature(amr, map_data, config))
    obs = np.concatenate(blocks).astype(np.float32)
    if obs.shape[0] != config.obs_dim:
        raise ValueError(f"observation is {obs.shape[0]}-dim, config says "
                         f"{config.obs_dim}; call V1Config.validate()")
    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)


def build_cluster_observation_np(cluster, worker_predictions, map_data,
                                 config) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Padded cluster observation as numpy.

    Returns ``(obs [1, N_max, obs_dim], mask [1, N_max], amr_ids)``.
    ``mask[b, i] = 1`` marks a valid AMR slot.
    """
    members = list(cluster)[:config.max_agents]
    obs = np.zeros((1, config.max_agents, config.obs_dim), dtype=np.float32)
    mask = np.zeros((1, config.max_agents), dtype=np.float32)
    ids: list[str] = []
    for i, amr in enumerate(members):
        obs[0, i] = build_amr_observation(amr, members, worker_predictions,
                                          map_data, config)
        mask[0, i] = 1.0
        ids.append(amr.id)
    return obs, mask, ids


def build_cluster_observation(cluster, worker_predictions, map_data, config):
    """Torch version used by the online planner (spec interface)."""
    import torch
    obs, mask, ids = build_cluster_observation_np(
        cluster, worker_predictions, map_data, config)
    return (torch.as_tensor(obs), torch.as_tensor(mask), ids)
