from __future__ import annotations

import numpy as np

try:
    from .path import obs_normalization_scales
except ImportError:
    from path import obs_normalization_scales


def obs_to_path_goal(obs, cfg, lookahead_idx: int = 2) -> np.ndarray:
    """Extract the i-th lookahead path point from obs as an action."""
    base = 4 + (lookahead_idx - 1) * 2
    lx, ly = float(obs[base]), float(obs[base + 1])

    if cfg.get("normalize_obs", False):
        _, pos_s, _, _, _ = obs_normalization_scales(cfg)
        lx, ly = lx * pos_s, ly * pos_s

    return np.array([lx, ly], dtype=np.float32)


class HybridPolicy:
    """State machine that switches between path following and RL avoidance."""

    FOLLOW_PATH = "follow_path"
    RL_AVOID = "rl_avoid"

    def __init__(
        self,
        rl_model,
        env,
        *,
        lat_thresh: float | None = None,
        hdg_thresh: float | None = None,
        follow_lookahead: int = 3,
    ):
        self.rl_model = rl_model
        self.env = env
        self.cfg = env.cfg
        self.state = self.FOLLOW_PATH

        self.lat_thresh = lat_thresh or float(self.cfg["success_lat_thresh"])
        self.hdg_thresh = hdg_thresh or float(self.cfg["success_hdg_thresh"])
        self.follow_lookahead = follow_lookahead

        self._normalize = self.cfg.get("normalize_obs", False)
        if self._normalize:
            scales = obs_normalization_scales(self.cfg)
            self._lat_scale = scales[0]
        else:
            self._lat_scale = 1.0

    def reset(self) -> None:
        self.state = self.FOLLOW_PATH

    def _decode_obs(self, obs) -> tuple[float, float, float]:
        lat_raw = float(obs[2])
        hdg_raw = float(obs[3])
        risk = float(obs[-1])

        if self._normalize:
            lat = abs(lat_raw * self._lat_scale)
            hdg = abs(hdg_raw * np.pi)
        else:
            lat = abs(lat_raw)
            hdg = abs(hdg_raw)

        return lat, hdg, risk

    def predict(self, obs, deterministic: bool = True):
        lat, hdg, risk = self._decode_obs(obs)

        if self.state == self.FOLLOW_PATH:
            if risk > 0.5:
                self.state = self.RL_AVOID
        elif self.state == self.RL_AVOID:
            human_clear = risk < 0.5
            if human_clear:
                self.state = self.FOLLOW_PATH

        if self.state == self.RL_AVOID:
            action, _ = self.rl_model.predict(obs, deterministic=deterministic)
        else:
            action = obs_to_path_goal(obs, self.cfg, lookahead_idx=self.follow_lookahead)

        return action, self.state


def random_forward_action(env) -> np.ndarray:
    a = env.action_space.sample()
    a[0] = np.clip(a[0] + 1.0, *env.cfg["goal_fwd_range"])
    a[1] *= 0.3
    return a.astype(np.float32)


def follow_path_action(obs, env, lookahead_idx: int = 2) -> np.ndarray:
    return obs_to_path_goal(obs, env.cfg, lookahead_idx=lookahead_idx)


def reactive_avoid_action(obs, env) -> np.ndarray:
    cfg = env.cfg
    action = obs_to_path_goal(obs, cfg, lookahead_idx=2)

    n_lk = cfg["n_lookahead"]
    h_rx = obs[4 + 2 * n_lk]
    h_ry = obs[4 + 2 * n_lk + 1]
    dist = np.hypot(h_rx, h_ry)

    if dist < cfg["safety_dist"] * 2.0 and h_rx > 0:
        steer = -np.sign(h_ry + 1e-3) * 1.5
        action[1] = np.clip(action[1] + steer, *cfg["goal_lat_range"])
        action[0] = np.clip(action[0] * 0.6, *cfg["goal_fwd_range"])

    return action.astype(np.float32)


def dodge_behind_action(obs, env) -> np.ndarray:
    cfg = env.cfg
    n_lk = cfg["n_lookahead"]
    detect_radius = cfg["safety_dist"] * 2.5
    dodge_strength = 1.5

    h_rx = obs[4 + 2 * n_lk]
    h_ry = obs[4 + 2 * n_lk + 1]
    h_dist = np.hypot(h_rx, h_ry)

    human_ahead = h_rx > 0
    human_close = h_dist < detect_radius

    if human_ahead and human_close:
        action = obs_to_path_goal(obs, cfg, lookahead_idx=2)
        cr = np.cos(env.rtheta)
        sr = np.sin(env.rtheta)
        h_vy_rf = -sr * env.hvx + cr * env.hvy

        dodge_dir = -np.sign(h_vy_rf + 1e-6)
        intensity = np.clip(1.0 - h_dist / detect_radius, 0.0, 1.0)

        action[1] = np.clip(
            action[1] + dodge_dir * dodge_strength * intensity,
            *cfg["goal_lat_range"],
        )
        action[0] = np.clip(action[0] * 0.7, *cfg["goal_fwd_range"])
    else:
        action = obs_to_path_goal(obs, cfg, lookahead_idx=4)
        lat_offset = obs[2]
        action[1] = np.clip(action[1] - 1.5 * lat_offset, *cfg["goal_lat_range"])

    return action.astype(np.float32)


def stop_and_wait_action(obs, env) -> np.ndarray:
    cfg = env.cfg
    n_lk = cfg["n_lookahead"]
    detect_radius = cfg["safety_dist"] * 2.5
    lane_half_w = cfg["corridor_w"] * 1.5

    h_rx = obs[4 + 2 * n_lk]
    h_ry = obs[4 + 2 * n_lk + 1]
    h_dist = np.hypot(h_rx, h_ry)

    human_ahead = h_rx > 0
    human_close = h_dist < detect_radius
    human_in_lane = abs(h_ry) < lane_half_w

    if human_ahead and human_close and human_in_lane:
        return np.array([0.0, 0.0], dtype=np.float32)

    return obs_to_path_goal(obs, cfg, lookahead_idx=2)


def slow_on_path_action(obs, env) -> np.ndarray:
    cfg = env.cfg
    n_lk = cfg["n_lookahead"]
    detect_radius = cfg["safety_dist"] * 2.5
    lane_half_w = cfg["corridor_w"] * 1.5
    min_speed_ratio = 0.15

    h_rx = obs[4 + 2 * n_lk]
    h_ry = obs[4 + 2 * n_lk + 1]
    h_dist = np.hypot(h_rx, h_ry)

    human_ahead = h_rx > 0
    human_close = h_dist < detect_radius
    human_in_lane = abs(h_ry) < lane_half_w

    action = obs_to_path_goal(obs, cfg, lookahead_idx=2)

    if human_ahead and human_close and human_in_lane:
        speed_ratio = min_speed_ratio + (1.0 - min_speed_ratio) * (h_dist / detect_radius)
        action[0] = np.clip(action[0] * speed_ratio, *cfg["goal_fwd_range"])

    return action.astype(np.float32)