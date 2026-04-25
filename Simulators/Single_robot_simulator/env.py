from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from .controller import PurePursuitController
    from .path import ReferencePath, obs_normalization_scales, wrap_angle
    from .rendering import close_render, render_env, start_recording, stop_recording
    from .reward import compute_reward, compute_reward_terms
except ImportError:
    from controller import PurePursuitController
    from path import ReferencePath, obs_normalization_scales, wrap_angle
    from rendering import close_render, render_env, start_recording, stop_recording
    from reward import compute_reward, compute_reward_terms


class LocalPlannerEnv(gym.Env):
    """Gymnasium environment for local path revision around a dynamic human."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    DEFAULT_CFG: dict = dict(
        map_size=20.0,
        dt=0.1,
        max_steps=200,
        robot_radius=0.3,
        max_v=1.0,
        max_omega=1.0,
        init_v=0.6,
        human_radius=0.3,
        human_speed_range=(0.1, 0.3),
        collision_dist=0.6,
        safety_dist=1.5,
        corridor_len=8.0,
        corridor_w=1.8,
        goal_fwd_range=(0.0, 3.0),
        goal_lat_range=(-2.0, 2.0),
        n_lookahead=8,
        lookahead_spacing=1.0,
        w_collision=-200.0,
        w_safety=-5.0,
        w_deviation=-10,
        w_heading=-2.0,
        w_progress=40,
        w_speed=2.0,
        w_time=-0.5,
        w_success=100,
        path_pen_min=0.15,
        path_pen_restore_dist=3,
        success_lat_thresh=0.05,
        success_hdg_thresh=0.1,
        oob_margin=2.0,
        human_delay=2.0,
        p_ambient_human=0.0,
        normalize_obs=True,
        return_reward_breakdown=False,
    )

    def __init__(self, config: dict | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = dict(self.DEFAULT_CFG)
        if config:
            self.cfg.update(config)
        self.render_mode = render_mode

        c = self.cfg
        n_lk = c["n_lookahead"]
        obs_dim = 1 + 3 + 2 * n_lk + 4 + 1

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([c["goal_fwd_range"][0], c["goal_lat_range"][0]], dtype=np.float32),
            high=np.array([c["goal_fwd_range"][1], c["goal_lat_range"][1]], dtype=np.float32),
        )

        self.controller = PurePursuitController(c["max_v"], c["max_omega"])

        self.path: ReferencePath | None = None
        self.rx = self.ry = self.rtheta = self.rv = 0.0
        self.hx = self.hy = self.hvx = self.hvy = 0.0
        self.cur_s = 0.0
        self.steps = 0
        self._h_behav = "cross"

        self._rtraj: list[np.ndarray] = []
        self._htraj: list[np.ndarray] = []
        self._goals: list[np.ndarray] = []

        self._human_visible = False
        self._human_appear_step = int(round(self.cfg["human_delay"] / self.cfg["dt"]))
        self._ep_min_d_human = float("inf")
        self._prev_abs_lat = 0.0

        self._fig = None
        self._ax = None
        self._recording = False
        self._frames: list[np.ndarray] = []

    # ------------------------
    # Recording / rendering
    # ------------------------

    def start_recording(self) -> None:
        start_recording(self)

    def stop_recording(self, path: str = "episode.mp4", fps: int = 10) -> str | None:
        return stop_recording(self, path, fps)

    def render(self):
        return render_env(self)

    def close(self):
        close_render(self)

    # ------------------------
    # Internal helpers
    # ------------------------

    def _make_path(self, rng: np.random.Generator) -> None:
        ms, mg = self.cfg["map_size"], 3.0
        x_start, x_end = mg - 1.0, ms - mg
        y_mid = ms / 2.0

        n_pts = 20
        xs = np.linspace(x_start, x_end, n_pts)
        ys = np.full_like(xs, y_mid)
        self.path = ReferencePath(np.column_stack([xs, ys]))

    def _spawn_human_ambient(self, rng: np.random.Generator) -> None:
        c = self.cfg
        side = float(rng.choice([-1.0, 1.0]))
        s0 = self.cur_s + float(rng.uniform(4.0, 11.0))
        s0 = float(np.clip(s0, self.cur_s + 2.0, self.path.total_length - 2.0))
        p0 = self.path.position(s0)
        n0 = self.path.normal(s0)
        tan0 = self.path.tangent(s0)
        lateral_min = c["corridor_w"] + float(rng.uniform(0.7, 1.8))
        lateral_max = min(float(c["map_size"]) * 0.28, lateral_min + 3.5)
        lateral = float(rng.uniform(lateral_min, max(lateral_max, lateral_min + 0.5)))
        start = p0 + side * n0 * lateral
        v_walk = float(rng.uniform(0.05, 0.17))
        self.hx, self.hy = float(start[0]), float(start[1])
        self.hvx, self.hvy = float(tan0[0] * v_walk), float(tan0[1] * v_walk)
        self._h_behav = "ambient"

    def _spawn_human(self, rng: np.random.Generator) -> None:
        c = self.cfg
        if rng.random() < float(c.get("p_ambient_human", 0.0)):
            self._spawn_human_ambient(rng)
            return

        behav = str(rng.choice(["cross", "side", "along"]))
        speed = float(rng.uniform(*c["human_speed_range"]))

        v_r = (c["init_v"] + c["max_v"]) * 0.5

        t_lo, t_hi = c.get("encounter_t_range", (3.0, 6.0))
        t_enc = float(rng.uniform(t_lo, t_hi))

        enc_s = self.cur_s + v_r * t_enc
        enc_s = min(enc_s, self.path.total_length - 2.0)
        enc_pos = self.path.position(enc_s)
        nrm = self.path.normal(enc_s)
        p_from_below = float(c.get("human_from_below_prob", 0.5))
        side = -1.0 if rng.random() < p_from_below else 1.0

        jitter_lo, jitter_hi = c.get("encounter_jitter", (0.85, 1.1))
        t_h = t_enc * float(rng.uniform(jitter_lo, jitter_hi))

        if behav == "cross":
            start_dist = speed * t_h
            start = enc_pos + side * nrm * start_dist
            d = -side * nrm
            vx, vy = float(d[0] * speed), float(d[1] * speed)

        elif behav == "side":
            ang = float(rng.uniform(0.3, 0.7))
            ca, sa = np.cos(ang), np.sin(ang)
            d = -side * nrm
            d = np.array([d[0] * ca - d[1] * sa, d[0] * sa + d[1] * ca])
            start_dist = speed * t_h
            start = enc_pos - d * start_dist
            vx, vy = float(d[0] * speed), float(d[1] * speed)

        else:
            slow = speed * 0.4
            gap = (v_r - slow) * t_enc
            h_s0 = self.cur_s + gap
            h_s0 = float(np.clip(h_s0, self.cur_s + 1.0, self.path.total_length - 1.0))
            p0 = self.path.position(h_s0)
            n0 = self.path.normal(h_s0)
            start = p0 + side * n0 * float(rng.uniform(0.1, 0.3))
            t0 = self.path.tangent(h_s0)
            vx, vy = float(t0[0] * slow), float(t0[1] * slow)

        self.hx, self.hy = float(start[0]), float(start[1])
        self.hvx, self.hvy = vx, vy
        self._h_behav = behav

    def _in_corridor(self, px: float, py: float) -> bool:
        c = self.cfg
        s_lo = self.cur_s
        s_hi = min(self.cur_s + c["corridor_len"], self.path.total_length)
        mid = (s_lo + s_hi) / 2
        rad = (s_hi - s_lo) / 2 + 1.0
        s_cl, _, dist = self.path.closest_point(px, py, s_hint=mid, search_radius=rad)
        if s_cl < s_lo - 0.3 or s_cl > s_hi + 0.3:
            return False
        return dist < c["corridor_w"]

    def _activate_human(self) -> None:
        self._spawn_human(self.np_random)
        self._human_visible = True

    def _update_progress(self) -> None:
        s_new, _, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
        )
        self.cur_s = max(self.cur_s, s_new)

    # ------------------------
    # Observation
    # ------------------------

    def _obs(self) -> np.ndarray:
        c = self.cfg
        s = self.cur_s
        _, lat, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=s, search_radius=5.0,
        )
        h_err = wrap_angle(self.rtheta - self.path.heading(s))
        progress = s / self.path.total_length

        cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)

        look: list[float] = []
        for i in range(1, c["n_lookahead"] + 1):
            sa = min(s + i * c["lookahead_spacing"], self.path.total_length)
            p = self.path.position(sa)
            dx, dy = p[0] - self.rx, p[1] - self.ry
            look.extend([cr * dx + sr * dy, -sr * dx + cr * dy])

        if self._human_visible:
            dx, dy = self.hx - self.rx, self.hy - self.ry
            hrx = cr * dx + sr * dy
            hry = -sr * dx + cr * dy
            dvx = self.hvx - self.rv * np.cos(self.rtheta)
            dvy = self.hvy - self.rv * np.sin(self.rtheta)
            hrvx = cr * dvx + sr * dvy
            hrvy = -sr * dvx + cr * dvy
            risk = 1.0 if self._in_corridor(self.hx, self.hy) else 0.0
        else:
            hrx, hry, hrvx, hrvy, risk = 10.0, 0.0, 0.0, 0.0, 0.0

        vec = [self.rv, progress, lat, h_err] + look + [hrx, hry, hrvx, hrvy, risk]

        if c.get("normalize_obs", False):
            lat_s, pos_s, vel_s, ms, mv = obs_normalization_scales(c)
            vec[0] = float(vec[0]) / mv
            vec[2] = float(vec[2]) / lat_s
            vec[3] = float(vec[3]) / np.pi

            for i in range(4, 4 + 2 * c["n_lookahead"], 2):
                vec[i] = float(vec[i]) / pos_s
                vec[i + 1] = float(vec[i + 1]) / pos_s

            hb = 4 + 2 * c["n_lookahead"]
            vec[hb] = float(vec[hb]) / ms
            vec[hb + 1] = float(vec[hb + 1]) / ms
            vec[hb + 2] = float(vec[hb + 2]) / vel_s
            vec[hb + 3] = float(vec[hb + 3]) / vel_s

        return np.asarray(vec, dtype=np.float32)

    # ------------------------
    # Reward
    # ------------------------

    def _reward_terms(self, old_s: float, collision: bool, success: bool) -> dict[str, float]:
        return compute_reward_terms(self, old_s, collision, success)

    def _reward(self, old_s: float, collision: bool, success: bool) -> float:
        return compute_reward(self, old_s, collision, success)

    # ------------------------
    # Termination
    # ------------------------

    def _check_done(self) -> tuple[bool, bool, bool, bool, dict]:
        c = self.cfg
        info: dict = {}
        terminated = truncated = collision = success = False

        if self._human_visible:
            dh = np.hypot(self.rx - self.hx, self.ry - self.hy)
            if dh < c["collision_dist"]:
                terminated, collision = True, True
                info["collision"] = True
                return terminated, truncated, collision, success, info

            _, lat, _ = self.path.closest_point(
                self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
            )
            h_err = abs(wrap_angle(self.rtheta - self.path.heading(self.cur_s)))
            on_path = abs(lat) < c["success_lat_thresh"] and h_err < c["success_hdg_thresh"]

            cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
            h_ahead = cr * (self.hx - self.rx) + sr * (self.hy - self.ry)
            human_behind = h_ahead < 0
            human_far = dh > c["safety_dist"] * 2
            human_clear = (not self._in_corridor(self.hx, self.hy)) and (human_far or human_behind)
        else:
            on_path = False
            human_clear = False

        if on_path and human_clear and self.steps > self._human_appear_step + 10:
            terminated, success = True, True
            info["success"] = True
            return terminated, truncated, collision, success, info

        if self.steps >= c["max_steps"]:
            truncated = True
            info["timeout"] = True

        m, ms = c["oob_margin"], c["map_size"]
        if self.rx < -m or self.rx > ms + m or self.ry < -m or self.ry > ms + m:
            truncated = True
            info["out_of_bounds"] = True

        if self.cur_s >= self.path.total_length - 1.0:
            truncated = True
            info["path_end"] = True

        return terminated, truncated, collision, success, info

    # ------------------------
    # Gym API
    # ------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        self._make_path(rng)
        self.cur_s = float(rng.uniform(0.3, 1.5))

        p = self.path.position(self.cur_s)
        h = self.path.heading(self.cur_s)
        self.rx, self.ry, self.rtheta = float(p[0]), float(p[1]), h
        self.rv = 0.0

        self._human_visible = False
        self._human_appear_step = int(round(self.cfg["human_delay"] / self.cfg["dt"]))
        self.hx, self.hy, self.hvx, self.hvy = 0.0, 0.0, 0.0, 0.0
        self._h_behav = ""

        self.steps = 0
        self._ep_min_d_human = float("inf")
        self._prev_abs_lat = 0.0
        self._rtraj = [np.array([self.rx, self.ry])]
        self._htraj = []
        self._goals = []

        obs = self._obs()
        info = {"behavior": "pending"}
        return obs, info

    def step(self, action):
        c = self.cfg
        self.steps += 1
        old_s = self.cur_s

        if not self._human_visible and self.steps >= self._human_appear_step:
            self._activate_human()

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        fwd, lat = float(action[0]), float(action[1])

        cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
        goal = np.array([
            self.rx + fwd * cr - lat * sr,
            self.ry + fwd * sr + lat * cr,
        ])
        self._goals.append(goal.copy())

        if abs(fwd) < 0.05 and abs(lat) < 0.05:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            v_cmd, w_cmd = self.controller.compute(self.rx, self.ry, self.rtheta, goal)

        dt = c["dt"]
        v = float(np.clip(v_cmd, 0, c["max_v"]))
        w = float(np.clip(w_cmd, -c["max_omega"], c["max_omega"]))

        self.rx += v * np.cos(self.rtheta) * dt
        self.ry += v * np.sin(self.rtheta) * dt
        self.rtheta = wrap_angle(self.rtheta + w * dt)
        self.rv = v

        if self._human_visible:
            self.hx += self.hvx * dt
            self.hy += self.hvy * dt
            self._htraj.append(np.array([self.hx, self.hy]))
            dh_step = float(np.hypot(self.rx - self.hx, self.ry - self.hy))
            self._ep_min_d_human = min(self._ep_min_d_human, dh_step)

        self._rtraj.append(np.array([self.rx, self.ry]))

        self._update_progress()
        terminated, truncated, collision, success, info = self._check_done()
        reward = self._reward(old_s, collision, success)
        obs = self._obs()

        info["step"] = self.steps
        if self._human_visible:
            info["behavior"] = self._h_behav

        if c.get("return_reward_breakdown"):
            info["reward_terms"] = self._reward_terms(old_s, collision, success)

        if terminated or truncated:
            _, lat_f, _ = self.path.closest_point(
                self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
            )
            h_err_f = abs(wrap_angle(self.rtheta - self.path.heading(self.cur_s)))
            on_path_end = (
                abs(lat_f) < c["success_lat_thresh"]
                and h_err_f < c["success_hdg_thresh"]
            )

            ep_min = (
                float(self._ep_min_d_human)
                if self._ep_min_d_human < float("inf")
                else -1.0
            )

            human_clear_end = False
            if self._human_visible:
                dh_e = float(np.hypot(self.rx - self.hx, self.ry - self.hy))
                cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
                h_ahead = cr * (self.hx - self.rx) + sr * (self.hy - self.ry)
                human_behind = h_ahead < 0
                human_far = dh_e > c["safety_dist"] * 2
                human_clear_end = (
                    not self._in_corridor(self.hx, self.hy)
                    and (human_far or human_behind)
                )

            info["episode_stats"] = {
                "collision": bool(collision),
                "success": bool(success),
                "min_human_dist": ep_min,
                "final_abs_lateral": float(abs(lat_f)),
                "on_path_at_end": bool(on_path_end),
                "human_clear_at_end": bool(human_clear_end),
            }

        return obs, reward, terminated, truncated, info