from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from .controller import PurePursuitController
    from .path import ReferencePath, obs_normalization_scales, wrap_angle
    from .policies import obs_to_path_goal
    from .rendering import close_render, render_env, start_recording, stop_recording
    from .reward import compute_reward, compute_reward_terms
except ImportError:
    from controller import PurePursuitController
    from path import ReferencePath, obs_normalization_scales, wrap_angle
    from policies import obs_to_path_goal
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
        collision_dist=0.7,
        safety_dist=1.5,
        corridor_len=8.0,
        corridor_w=1.8,
        goal_fwd_range=(0.0, 3.0),
        goal_lat_range=(-2.0, 2.0),
        n_lookahead=8,
        lookahead_spacing=1.0,
        w_collision=-200.0,
        w_safety=-5.0,
        w_deviation=-10, #decrease: -8
        w_heading=-2.0, #maybe decrease: -1.5
        w_progress=40, #increase: 50
        w_speed=2.0,
        w_time=-0.5,
        w_success=100,
        path_pen_min=0.15,
        path_pen_restore_dist=3,
        success_lat_thresh=0.05,
        success_hdg_thresh=0.1,
        oob_margin=2.0,
        human_delay=2.0,
        human_exists_from_start=True,
        human_detection_len=8.0,
        human_detection_w=1.8,
        cross_s_range=(2.5, 5.5),
        cross_angle_jitter=0.45,
        cross_appear_dist_default=7.0,
        cross_time_scale=0.78,
        cross_start_dist_min=2.0,
        path_follow_min_dist=0.5,
        path_follow_max_dist=0.7,
        human_outside_corridor_margin=0.35,
        use_encounter_spawn=True,
        validate_blocking_encounters=True,
        encounter_validation="tune",
        encounter_validate_max_tries=10,
        encounter_accept_max_dist=0.75,
        encounter_reject_max_steps=55,
        encounter_reject_time_scale_samples=8,
        encounter_reject_appear_samples=4,
        encounter_accept_min_dist=None,
        encounter_tune_attempts=3,
        encounter_tune_max_steps=80,
        use_state_machine=True,
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
        self._gtraj: list[np.ndarray] = []
        self._htraj: list[np.ndarray] = []
        self._goals: list[np.ndarray] = []

        self._human_visible = False
        self._human_exists = False
        self._human_observable = False
        self._rl_active = False
        self._rl_latched = False
        self._encounter_active = False
        self._human_appear_step = int(round(self.cfg["human_delay"] / self.cfg["dt"]))
        self._ep_min_d_human = float("inf")
        self._ep_min_d_path_following = float("inf")
        self._pending_encounter: dict | None = None
        self._active_cross_s = 0.0
        self._ghost_rx = self._ghost_ry = self._ghost_rtheta = self._ghost_rv = 0.0
        self._ghost_cur_s = 0.0
        self._ghost_visible = False
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
        self._human_exists = True
        self._human_observable = True
        self._human_visible = True
        self._encounter_active = False

    def _in_detection_zone(self, px: float, py: float) -> bool:
        c = self.cfg
        s_lo = self.cur_s
        s_hi = min(
            self.cur_s + c.get("human_detection_len", c["corridor_len"]),
            self.path.total_length,
        )
        mid = (s_lo + s_hi) / 2
        rad = (s_hi - s_lo) / 2 + 1.0
        s_cl, _, dist = self.path.closest_point(px, py, s_hint=mid, search_radius=rad)
        if s_cl < s_lo - 0.3 or s_cl > s_hi + 0.3:
            return False
        return dist < c.get("human_detection_w", c["corridor_w"])

    def _path_follow_action(self, lookahead_idx: int = 3) -> np.ndarray:
        c = self.cfg
        lookahead_s = min(
            self.cur_s + lookahead_idx * c["lookahead_spacing"],
            self.path.total_length,
        )
        target = self.path.position(lookahead_s)
        dx = target[0] - self.rx
        dy = target[1] - self.ry
        cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
        fwd = cr * dx + sr * dy
        lat = -sr * dx + cr * dy
        action = np.array([fwd, lat], dtype=np.float32)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _lateral_dist_to_path(self, px: float, py: float, s_hint: float) -> float:
        _, _, dist = self.path.closest_point(px, py, s_hint=s_hint, search_radius=8.0)
        return float(dist)

    def _build_random_encounter(self, rng: np.random.Generator) -> dict:
        """Build a draft crossing encounter dict (full-run style parameters)."""
        c = self.cfg
        speed = float(rng.uniform(*c["human_speed_range"]))
        s_lo, s_hi = c.get("cross_s_range", (2.5, 5.5))
        s_enc = self.cur_s + float(rng.uniform(s_lo, s_hi))
        s_enc = min(s_enc, self.path.total_length - 2.0)
        side = -1.0 if rng.random() < c.get("human_from_below_prob", 0.5) else 1.0
        jitter_max = float(c.get("cross_angle_jitter", 0.45))
        cross_angle = float(rng.uniform(-jitter_max, jitter_max))
        return {
            "s": float(s_enc),
            "behavior": "cross",
            "speed": speed,
            "side": side,
            "cross_angle": cross_angle,
            "appear_distance": float(c.get("cross_appear_dist_default", 7.0)),
            "time_scale": float(c.get("cross_time_scale", 0.78)),
        }

    def _encounter_validation_mode(self) -> str:
        """How encounters are validated on reset: ``none``, ``reject``, or ``tune``."""
        mode = self.cfg.get("encounter_validation")
        if mode is not None:
            return str(mode).lower()
        legacy = self.cfg.get("validate_blocking_encounters", True)
        if legacy == "reject":
            return "reject"
        if legacy:
            return "tune"
        return "none"

    def _reject_sample_blocking_encounter(self, rng: np.random.Generator) -> dict:
        """Resample random encounters until a short path-only check passes."""
        c = self.cfg
        max_tries = int(c.get("encounter_validate_max_tries", 10))
        accept_max = float(c.get("encounter_accept_max_dist", 0.75))
        accept_min = c.get("encounter_accept_min_dist")
        if accept_min is not None:
            accept_min = float(accept_min)
        tune_seed = int(rng.integers(0, 2**31 - 1))
        ts_samples = int(c.get("encounter_reject_time_scale_samples", 8))

        best: dict | None = None
        best_m = float("inf")

        for attempt in range(max_tries):
            draft = self._build_random_encounter(rng)
            fast = _fast_time_scale_search(
                draft,
                c,
                seed=tune_seed + attempt * 100,
                start_cur_s=self.cur_s,
                accept_max=accept_max,
                accept_min=accept_min,
                n_samples=ts_samples,
            )
            if fast is not None:
                return fast
            result = simulate_path_only_encounter(
                draft,
                {**c, "encounter_tune_max_steps": c.get("encounter_reject_max_steps", 55),
                 "validate_blocking_encounters": False},
                seed=tune_seed + attempt,
                start_cur_s=self.cur_s,
            )
            m = float(result["min_dist_path_following"])
            if m < best_m:
                best_m = m
                best = draft

        if best is not None:
            return best

        lo = accept_min if accept_min is not None else 0.0
        raise RuntimeError(
            f"Could not sample a blocking encounter in {max_tries} reject attempts "
            f"(accept: {lo} <= min_dist < {accept_max} m, best seen {best_m:.2f} m)"
        )

    def _build_and_tune_random_encounter(self, rng: np.random.Generator) -> dict:
        """Build a crossing encounter; optionally validate via reject sampling or tuning."""
        mode = self._encounter_validation_mode()
        if mode == "none":
            return self._build_random_encounter(rng)
        if mode == "reject":
            return self._reject_sample_blocking_encounter(rng)

        max_attempts = int(self.cfg.get("encounter_tune_attempts", 3))
        tune_seed = int(rng.integers(0, 2**31 - 1))
        angle_max = float(self.cfg.get("cross_angle_jitter", 0.45))

        last_draft: dict | None = None
        carry: dict | None = None
        for attempt in range(max_attempts):
            draft = carry if carry is not None else self._build_random_encounter(rng)
            carry = None
            last_draft = draft
            tuned = tune_blocking_encounter(
                draft,
                self.cfg,
                seed=tune_seed + attempt * 97,
                start_cur_s=self.cur_s,
            )
            if tuned is not None:
                return tuned

            carry = dict(draft)
            carry["s"] = max(
                self.cur_s + 2.0,
                float(carry["s"]) - float(rng.uniform(0.3, 0.8)),
            )
            carry["speed"] = min(
                float(self.cfg["human_speed_range"][1]),
                float(carry["speed"]) + float(rng.uniform(0.02, 0.06)),
            )
            carry["cross_angle"] = float(rng.uniform(-angle_max, angle_max))

        speed_hi = float(self.cfg["human_speed_range"][1])
        s_candidates = np.linspace(self.cur_s + 1.8, self.cur_s + 6.0, 8)
        for s_enc in s_candidates:
            s_val = float(np.clip(s_enc, self.cur_s + 1.2, self.path.total_length - 2.0))
            for side in (1.0, -1.0):
                draft = {
                    "s": s_val,
                    "behavior": "cross",
                    "speed": speed_hi,
                    "side": float(side),
                    "cross_angle": float(rng.uniform(-angle_max, angle_max)),
                    "appear_distance": float(self.cfg.get("cross_appear_dist_default", 7.0)),
                    "time_scale": float(self.cfg.get("cross_time_scale", 0.78)),
                }
                tuned = tune_blocking_encounter(
                    draft,
                    self.cfg,
                    seed=tune_seed + int(1000 * s_val) + int(side * 10),
                    start_cur_s=self.cur_s,
                )
                if tuned is not None:
                    return tuned

        raise RuntimeError(
            "Could not tune a blocking encounter for the single-robot scene "
            f"(path_follow_dist=[{self.cfg.get('path_follow_min_dist', 0.5)}, "
            f"{self.cfg.get('path_follow_max_dist', 0.7)}])"
        )

    def _spawn_encounter_human(self, enc: dict, cross_s: float | None = None) -> None:
        """Spawn a pedestrian timed to cross the reference path ahead of the robot."""
        c = self.cfg
        s_enc = float(cross_s if cross_s is not None else enc["s"])
        speed = float(enc.get("speed", 0.2))
        side = float(enc.get("side", 1.0))

        enc_pos = self.path.position(s_enc)
        nrm = self.path.normal(s_enc)

        if "cross_angle" in enc:
            ang = float(enc["cross_angle"])
        else:
            jitter_max = float(c.get("cross_angle_jitter", 0.45))
            ang = float(self.np_random.uniform(-jitter_max, jitter_max))

        gap = max(s_enc - self.cur_s, 0.5)
        v_r = max(self.rv, c.get("init_v", 0.6))
        time_scale = float(enc.get("time_scale", c.get("cross_time_scale", 0.78)))
        t_enc = gap / max(v_r, 0.1) * time_scale
        start_min = float(c.get("cross_start_dist_min", 2.0))
        start_dist = max(speed * t_enc, start_min)

        ca, sa = np.cos(ang), np.sin(ang)
        d_base = -side * nrm
        d = np.array([
            d_base[0] * ca - d_base[1] * sa,
            d_base[0] * sa + d_base[1] * ca,
        ])
        d = d / max(float(np.linalg.norm(d)), 1e-6)

        margin = float(c.get("human_outside_corridor_margin", 0.35))
        min_lat = float(c["corridor_w"]) + margin
        start_dist = max(speed * t_enc, start_min)
        for _ in range(40):
            start = enc_pos - d * start_dist
            if self._lateral_dist_to_path(float(start[0]), float(start[1]), s_enc) >= min_lat:
                break
            start_dist += 0.3

        start = enc_pos - d * start_dist
        self.hvx, self.hvy = float(d[0] * speed), float(d[1] * speed)
        self.hx, self.hy = float(start[0]), float(start[1])
        self._active_cross_s = s_enc
        self._h_behav = enc.get("behavior", "cross")
        self._human_exists = True
        self._human_visible = False
        self._encounter_active = False
        self._ep_min_d_human = float("inf")
        self._ep_min_d_path_following = float("inf")
        self._ghost_visible = False
        self._htraj = []

    def _maybe_spawn_human(self) -> None:
        if self._human_exists or self._pending_encounter is None:
            return

        enc = self._pending_encounter
        cross_s = float(enc["s"])
        appear_dist = float(
            enc.get("appear_distance", self.cfg.get("cross_appear_dist_default", 7.0))
        )
        if self.cur_s >= cross_s - appear_dist:
            self._spawn_encounter_human(enc, cross_s=cross_s)
            self._pending_encounter = None

    def _update_progress(self) -> None:
        s_new, _, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
        )
        self.cur_s = max(self.cur_s, s_new)

    def _obs_at(
        self,
        rx: float,
        ry: float,
        rtheta: float,
        rv: float,
        cur_s: float,
    ) -> np.ndarray:
        """Observation as if the robot were at the given state (shared human state)."""
        saved = (self.rx, self.ry, self.rtheta, self.rv, self.cur_s)
        self.rx, self.ry, self.rtheta, self.rv, self.cur_s = (
            rx, ry, rtheta, rv, cur_s,
        )
        obs = self._obs()
        self.rx, self.ry, self.rtheta, self.rv, self.cur_s = saved
        return obs

    def _step_path_follower(
        self,
        rx: float,
        ry: float,
        rtheta: float,
        rv: float,
        cur_s: float,
    ) -> tuple[float, float, float, float, float]:
        """Advance a virtual robot that only follows the reference path."""
        c = self.cfg
        obs = self._obs_at(rx, ry, rtheta, rv, cur_s)
        action = obs_to_path_goal(obs, c, lookahead_idx=3)
        fwd, lat = float(action[0]), float(action[1])

        cr, sr = np.cos(rtheta), np.sin(rtheta)
        goal = np.array([
            rx + fwd * cr - lat * sr,
            ry + fwd * sr + lat * cr,
        ])

        if abs(fwd) < 0.05 and abs(lat) < 0.05:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            v_cmd, w_cmd = self.controller.compute(rx, ry, rtheta, goal)

        dt = c["dt"]
        v = float(np.clip(v_cmd, 0, c["max_v"]))
        w = float(np.clip(w_cmd, -c["max_omega"], c["max_omega"]))
        rx = rx + v * np.cos(rtheta) * dt
        ry = ry + v * np.sin(rtheta) * dt
        rtheta = wrap_angle(rtheta + w * dt)
        s_new, _, _ = self.path.closest_point(
            rx, ry, s_hint=cur_s, search_radius=5.0,
        )
        cur_s = max(cur_s, s_new)
        return rx, ry, rtheta, v, cur_s

    def _is_on_path(self) -> bool:
        c = self.cfg
        _, lat, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
        )
        h_err = abs(wrap_angle(self.rtheta - self.path.heading(self.cur_s)))
        return (
            abs(lat) < c["success_lat_thresh"]
            and h_err < c["success_hdg_thresh"]
        )

    def _spawn_ghost_on_path(self) -> None:
        """Spawn ghost at the on-path position where the robot began deviating."""
        p = self.path.position(self.cur_s)
        h = self.path.heading(self.cur_s)
        self._ghost_rx = float(p[0])
        self._ghost_ry = float(p[1])
        self._ghost_rtheta = h
        self._ghost_rv = self.rv
        self._ghost_cur_s = self.cur_s
        self._ghost_visible = True
        self._gtraj = [np.array([self._ghost_rx, self._ghost_ry])]

    def _hide_ghost(self) -> None:
        self._ghost_visible = False

    def _update_ghost(self) -> None:
        """Show ghost only while robot is off-path during a human encounter."""
        if not self._human_visible:
            if self._ghost_visible:
                self._hide_ghost()
            return

        on_path = self._is_on_path()

        if not self._ghost_visible and not on_path:
            self._spawn_ghost_on_path()

        if self._ghost_visible:
            self._ghost_rx, self._ghost_ry, self._ghost_rtheta, self._ghost_rv, self._ghost_cur_s = (
                self._step_path_follower(
                    self._ghost_rx,
                    self._ghost_ry,
                    self._ghost_rtheta,
                    self._ghost_rv,
                    self._ghost_cur_s,
                )
            )
            self._gtraj.append(np.array([self._ghost_rx, self._ghost_ry]))
            d_ghost = float(np.hypot(
                self._ghost_rx - self.hx,
                self._ghost_ry - self.hy,
            ))
            self._ep_min_d_path_following = min(
                self._ep_min_d_path_following, d_ghost,
            )
            if on_path:
                self._hide_ghost()

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

        if self._human_exists:
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

        if self._encounter_active and on_path and human_clear and self.steps > self._human_appear_step + 10:
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

        self._human_appear_step = int(round(self.cfg["human_delay"] / self.cfg["dt"]))
        use_encounter_spawn = self.cfg.get("use_encounter_spawn", True)
        if self.cfg.get("human_exists_from_start", True):
            if use_encounter_spawn:
                forced = self.cfg.get("forced_encounter")
                if forced is not None:
                    self._pending_encounter = dict(forced)
                else:
                    self._pending_encounter = self._build_and_tune_random_encounter(rng)
                self._human_exists = False
                self._human_observable = False
                self._rl_active = False
                self._rl_latched = False
                self._encounter_active = False
                self.hx, self.hy, self.hvx, self.hvy = 0.0, 0.0, 0.0, 0.0
                self._h_behav = "cross"
            else:
                self._pending_encounter = None
                self._spawn_human(rng)
                self._human_exists = True
                self._human_observable = False
                self._rl_active = False
                self._rl_latched = False
                self._encounter_active = False
        else:
            self._pending_encounter = None
            self._human_exists = False
            self._human_observable = False
            self._rl_active = False
            self._rl_latched = False
            self._encounter_active = False
            self.hx, self.hy, self.hvx, self.hvy = 0.0, 0.0, 0.0, 0.0
            self._h_behav = ""

        self.steps = 0
        self._ep_min_d_human = float("inf")
        self._ep_min_d_path_following = float("inf")
        self._ghost_visible = False
        self._prev_abs_lat = 0.0
        self._rtraj = [np.array([self.rx, self.ry])]
        self._gtraj = []
        self._htraj = []
        self._goals = []
        self._human_visible = False

        obs = self._obs()
        info = {"behavior": self._h_behav if self._h_behav else "pending"}
        if self._pending_encounter is not None:
            info["encounter"] = dict(self._pending_encounter)
        return obs, info

    def step(self, action):
        c = self.cfg
        self.steps += 1
        old_s = self.cur_s

        if self.cfg.get("use_encounter_spawn", True):
            self._maybe_spawn_human()

        if not self.cfg.get("human_exists_from_start", True):
            if (not self._human_exists) and self.steps >= self._human_appear_step:
                self._activate_human()
                self._human_exists = True
                self._human_observable = True
                self._human_visible = True
                self._rl_active = True
                self._rl_latched = True

        raw_rl_action = np.asarray(action, dtype=np.float32)
        raw_rl_action = np.clip(raw_rl_action, self.action_space.low, self.action_space.high)
        if self.cfg.get("use_state_machine", True):
            if self._rl_active:
                action = raw_rl_action
            else:
                action = self._path_follow_action(lookahead_idx=3)
        else:
            action = raw_rl_action
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

        if self._human_exists:
            self.hx += self.hvx * dt
            self.hy += self.hvy * dt
            in_corridor = self._in_corridor(self.hx, self.hy)
            if in_corridor:
                self._encounter_active = True
            if self.cfg.get("use_encounter_spawn", True):
                if in_corridor and not self._human_visible:
                    self._human_visible = True
                    if self.cfg.get("use_state_machine", True):
                        self._rl_latched = True
                        self._rl_active = True
            else:
                self._human_observable = self._in_detection_zone(self.hx, self.hy)
                self._human_visible = self._human_observable or self._encounter_active
                if self._human_observable:
                    self._rl_latched = True
                self._rl_active = self._rl_latched
            if self._human_visible:
                self._htraj.append(np.array([self.hx, self.hy]))
                dh_step = float(np.hypot(self.rx - self.hx, self.ry - self.hy))
                self._ep_min_d_human = min(self._ep_min_d_human, dh_step)

        self._rtraj.append(np.array([self.rx, self.ry]))

        self._update_progress()
        self._update_ghost()
        terminated, truncated, collision, success, info = self._check_done()
        reward = self._reward(old_s, collision, success)
        obs = self._obs()

        info["step"] = self.steps
        if self._human_observable:
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
            ep_min_ghost = (
                float(self._ep_min_d_path_following)
                if self._ep_min_d_path_following < float("inf")
                else -1.0
            )

            human_clear_end = False
            if self._human_exists:
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
                "min_ghost_dist": ep_min_ghost,
                "min_dist_path_following": ep_min_ghost,
                "final_abs_lateral": float(abs(lat_f)),
                "on_path_at_end": bool(on_path_end),
                "human_clear_at_end": bool(human_clear_end),
            }

        return obs, reward, terminated, truncated, info


def _path_follow_dist_band(cfg: dict) -> tuple[float, float]:
    lo = float(cfg.get("path_follow_min_dist", 0.5))
    hi = float(cfg.get("path_follow_max_dist", 0.7))
    return lo, hi


def _encounter_dist_in_band(result: dict, cfg: dict) -> bool:
    lo, hi = _path_follow_dist_band(cfg)
    m = float(result.get("min_dist_path_following", float("inf")))
    if not (lo <= m < hi):
        return False
    return bool(result.get("human_in_corridor")) and bool(result.get("crosses_in_front"))


def _encounter_passes_reject(
    result: dict,
    *,
    accept_max: float,
    accept_min: float | None,
) -> bool:
    m = float(result.get("min_dist_path_following", float("inf")))
    if not (result.get("human_in_corridor") and result.get("crosses_in_front")):
        return False
    if m >= accept_max:
        return False
    if accept_min is not None and m < accept_min:
        return False
    return True


def _fast_time_scale_search(
    encounter: dict,
    cfg: dict,
    *,
    seed: int = 0,
    start_cur_s: float | None = None,
    accept_max: float = 0.75,
    accept_min: float | None = None,
    n_samples: int = 16,
) -> dict | None:
    """Coarse 1-D search over ``time_scale`` for a blocking encounter."""
    check_cfg = dict(cfg)
    check_cfg["validate_blocking_encounters"] = False
    reject_steps = int(cfg.get("encounter_reject_max_steps", 55))
    check_cfg["encounter_tune_max_steps"] = reject_steps

    best: dict | None = None
    best_m = float("inf")
    for j, ts in enumerate(np.linspace(0.25, 1.25, max(n_samples, 2))):
        candidate = {**encounter, "time_scale": float(ts)}
        result = simulate_path_only_encounter(
            candidate, check_cfg, seed=seed + j, start_cur_s=start_cur_s,
        )
        m = float(result["min_dist_path_following"])
        if m < best_m:
            best_m = m
            best = candidate
        if _encounter_passes_reject(
            result, accept_max=accept_max, accept_min=accept_min,
        ):
            return candidate
    if best is not None and best_m < accept_max:
        return best
    return None


def simulate_path_only_encounter(
    encounter: dict,
    config: dict | None = None,
    *,
    seed: int = 0,
    start_cur_s: float | None = None,
    lookahead_idx: int = 3,
) -> dict:
    """Run one encounter with pure path-following; check if it blocks the robot."""
    cfg = dict(LocalPlannerEnv.DEFAULT_CFG)
    if config:
        cfg.update(config)
    cfg["forced_encounter"] = dict(encounter)
    cfg["validate_blocking_encounters"] = False
    cfg["cross_angle_jitter"] = 0.0
    tune_steps = int(cfg.get("encounter_tune_max_steps", 80))
    cfg["max_steps"] = min(int(cfg.get("max_steps", 200)), tune_steps)

    env = LocalPlannerEnv(config=cfg)
    obs, _ = env.reset(seed=seed)
    if start_cur_s is not None:
        p = env.path.position(start_cur_s)
        h = env.path.heading(start_cur_s)
        env.cur_s = float(start_cur_s)
        env.rx, env.ry, env.rtheta = float(p[0]), float(p[1]), h

    done = False
    min_dist_corridor = float("inf")
    min_dist_ghost = float("inf")
    human_in_corridor = False
    crosses_in_front = False
    lo, hi = _path_follow_dist_band(cfg)
    enc_s = float(encounter["s"])
    info: dict = {}

    while not done:
        action = obs_to_path_goal(obs, env.cfg, lookahead_idx=lookahead_idx)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if env._human_exists and env._in_corridor(env.hx, env.hy):
            human_in_corridor = True
            dist = float(np.hypot(env.rx - env.hx, env.ry - env.hy))
            min_dist_corridor = min(min_dist_corridor, dist)
            cr, sr = np.cos(env.rtheta), np.sin(env.rtheta)
            h_ahead = cr * (env.hx - env.rx) + sr * (env.hy - env.ry)
            if h_ahead > 0.0:
                crosses_in_front = True

        if env._ghost_visible:
            d_ghost = float(np.hypot(env._ghost_rx - env.hx, env._ghost_ry - env.hy))
            min_dist_ghost = min(min_dist_ghost, d_ghost)

        if env._ep_min_d_path_following < float("inf"):
            min_dist_ghost = min(min_dist_ghost, env._ep_min_d_path_following)

        min_pf = min(min_dist_corridor, min_dist_ghost)
        if (
            info.get("collision")
            or info.get("success")
            or env.cur_s > enc_s + 5.0
            or (
                human_in_corridor
                and crosses_in_front
                and lo <= min_pf < hi
                and env.cur_s > enc_s + 0.5
            )
        ):
            done = True

    min_path_follow = min(min_dist_corridor, min_dist_ghost)
    blocking = (
        lo <= min_path_follow < hi
        and human_in_corridor
        and crosses_in_front
    )
    return {
        "blocking": blocking,
        "collision": bool(info.get("collision")),
        "human_in_corridor": human_in_corridor,
        "crosses_in_front": crosses_in_front,
        "min_dist_corridor": min_dist_corridor,
        "min_dist_ghost": min_dist_ghost,
        "min_dist_path_following": min_path_follow,
    }


def tune_blocking_encounter(
    enc: dict,
    cfg: dict,
    *,
    seed: int = 0,
    start_cur_s: float | None = None,
) -> dict | None:
    """Search timing so path-following min distance lands in [path_follow_min, path_follow_max]."""
    lo, hi = _path_follow_dist_band(cfg)
    target = 0.5 * (lo + hi)
    tune_cfg = dict(cfg)
    tune_cfg["encounter_tune_max_steps"] = cfg.get("encounter_tune_max_steps", 80)
    rng = np.random.default_rng(seed)
    s0 = float(enc.get("s", 4.0))
    speed0 = float(enc.get("speed", 0.25))
    side0 = float(enc.get("side", 1.0))
    ang0 = float(enc.get("cross_angle", 0.0))

    speed_lo, speed_hi = tune_cfg.get("human_speed_range", (0.1, 0.3))
    path_len = float(tune_cfg.get("map_size", 20.0)) - 2.0
    if start_cur_s is not None:
        s_lo = float(start_cur_s) + 1.0
    else:
        s_lo = max(0.5, s0 - 2.0)
    s_hi = min(path_len, s0 + 2.2)
    angle_lim = max(float(cfg.get("cross_angle_jitter", 0.45)), 0.45)

    best: dict | None = None
    best_err = float("inf")

    # Deterministic coarse pass near the provided draft.
    for appear in (3.0, 4.5, 6.0, 7.5, 9.0):
        for scale in (0.35, 0.50, 0.65, 0.80, 0.95):
            candidate = {
                **enc,
                "s": float(np.clip(s0, s_lo, s_hi)),
                "speed": float(np.clip(speed0, speed_lo, speed_hi)),
                "side": side0,
                "cross_angle": float(np.clip(ang0, -angle_lim, angle_lim)),
                "appear_distance": float(appear),
                "time_scale": float(scale),
            }
            result = simulate_path_only_encounter(
                candidate, tune_cfg, seed=seed, start_cur_s=start_cur_s,
            )
            if not _encounter_dist_in_band(result, cfg):
                continue
            err = abs(float(result["min_dist_path_following"]) - target)
            if err < best_err:
                best_err = err
                best = candidate
                if err < 0.01:
                    return best

    # Randomized pass to ensure angle and crossing variety.
    for _ in range(80):
        candidate = {
            **enc,
            "s": float(rng.uniform(s_lo, s_hi)),
            "speed": float(rng.uniform(speed_lo, speed_hi)),
            "side": float(side0 if rng.random() < 0.7 else -side0),
            "cross_angle": float(rng.uniform(-angle_lim, angle_lim)),
            "appear_distance": float(rng.uniform(2.5, 10.0)),
            "time_scale": float(rng.uniform(0.30, 1.20)),
        }
        result = simulate_path_only_encounter(
            candidate, tune_cfg, seed=seed, start_cur_s=start_cur_s,
        )
        if not _encounter_dist_in_band(result, cfg):
            continue
        err = abs(float(result["min_dist_path_following"]) - target)
        if err < best_err:
            best_err = err
            best = candidate
            if err < 0.01:
                return best

    return best
