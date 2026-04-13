"""
Lightweight 2D Simulator for RL-based Local Path Planning

Gymnasium-compatible environment for trajectory-conditioned local path revision.
A robot follows a reference path while avoiding a dynamic human who enters its
future path corridor. The policy outputs a local goal point which is tracked by
a pure-pursuit controller.

Usage:
    env = LocalPlannerEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
    env.close()
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


# ======================================================================
# Utility
# ======================================================================

def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ======================================================================
# Reference Path
# ======================================================================

class ReferencePath:
    """Smooth reference path built from waypoints with arc-length queries."""

    def __init__(self, waypoints: np.ndarray, num_samples: int = 1000):
        wp = np.asarray(waypoints, dtype=np.float64)
        t = np.linspace(0, 1, len(wp))
        self._cx = CubicSpline(t, wp[:, 0])
        self._cy = CubicSpline(t, wp[:, 1])

        t_f = np.linspace(0, 1, num_samples)
        self._px = self._cx(t_f)
        self._py = self._cy(t_f)

        ds = np.sqrt(np.diff(self._px) ** 2 + np.diff(self._py) ** 2)
        self._s = np.zeros(num_samples)
        self._s[1:] = np.cumsum(ds)
        self._t = t_f
        self.total_length: float = float(self._s[-1])

    def _s2t(self, s: float) -> float:
        return float(np.interp(np.clip(s, 0, self.total_length), self._s, self._t))

    def position(self, s: float) -> np.ndarray:
        t = self._s2t(s)
        return np.array([float(self._cx(t)), float(self._cy(t))])

    def tangent(self, s: float) -> np.ndarray:
        t = self._s2t(s)
        dx, dy = float(self._cx(t, 1)), float(self._cy(t, 1))
        norm = np.hypot(dx, dy) + 1e-9
        return np.array([dx / norm, dy / norm])

    def heading(self, s: float) -> float:
        t = self.tangent(s)
        return float(np.arctan2(t[1], t[0]))

    def normal(self, s: float) -> np.ndarray:
        """Left-pointing unit normal (90° counter-clockwise from tangent)."""
        t = self.tangent(s)
        return np.array([-t[1], t[0]])

    def closest_point(
        self, x: float, y: float,
        s_hint: float | None = None, search_radius: float = 5.0,
    ) -> tuple[float, float, float]:
        """Find the closest point on the path.

        Returns:
            (s, signed_lateral, unsigned_dist)
            signed_lateral > 0 means left of path.
        """
        if s_hint is not None:
            mask = np.abs(self._s - s_hint) <= search_radius
            if mask.sum() < 3:
                mask = np.ones(len(self._s), dtype=bool)
        else:
            mask = np.ones(len(self._s), dtype=bool)

        dx = self._px[mask] - x
        dy = self._py[mask] - y
        d2 = dx * dx + dy * dy
        idx = np.argmin(d2)

        s_cl = float(self._s[mask][idx])
        dist = float(np.sqrt(d2[idx]))

        nrm = self.normal(s_cl)
        p = self.position(s_cl)
        signed_lat = float(np.dot(np.array([x - p[0], y - p[1]]), nrm))

        return s_cl, signed_lat, dist

    def curvature(self, s: float) -> float:
        """Signed curvature κ at arc-length *s*."""
        t = self._s2t(s)
        dx = float(self._cx(t, 1))
        dy = float(self._cy(t, 1))
        ddx = float(self._cx(t, 2))
        ddy = float(self._cy(t, 2))
        denom = (dx * dx + dy * dy) ** 1.5 + 1e-12
        return (dx * ddy - dy * ddx) / denom

    def max_abs_curvature(self) -> float:
        """Return the maximum absolute curvature (vectorised)."""
        t_arr = self._t
        dx = self._cx(t_arr, 1)
        dy = self._cy(t_arr, 1)
        ddx = self._cx(t_arr, 2)
        ddy = self._cy(t_arr, 2)
        denom = (dx ** 2 + dy ** 2) ** 1.5 + 1e-12
        kappa = np.abs(dx * ddy - dy * ddx) / denom
        return float(np.max(kappa))

    def get_all_xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self._px.copy(), self._py.copy()


# ======================================================================
# Pure-Pursuit Controller
# ======================================================================

class PurePursuitController:
    """Converts a world-frame goal point to unicycle (v, ω) commands."""

    def __init__(self, max_v: float = 1.0, max_omega: float = 1.0):
        self.max_v = max_v
        self.max_omega = max_omega

    def compute(
        self, rx: float, ry: float, rtheta: float, goal: np.ndarray,
    ) -> tuple[float, float]:
        dx, dy = goal[0] - rx, goal[1] - ry
        c, s = np.cos(rtheta), np.sin(rtheta)
        lx = c * dx + s * dy
        ly = -s * dx + c * dy

        L = max(np.hypot(lx, ly), 0.3)
        kappa = 2.0 * ly / (L * L)

        v = self.max_v * float(np.clip(L / 2.0, 0.3, 1.0))
        v *= float(np.clip(1.0 - 0.5 * abs(kappa), 0.2, 1.0))

        omega = float(np.clip(v * kappa, -self.max_omega, self.max_omega))
        return v, omega


# ======================================================================
# Environment
# ======================================================================

class LocalPlannerEnv(gym.Env):
    """Gymnasium environment for local path revision around a dynamic human.

    Observation (float32 vector, dim = 1 + 3 + 2*N_look + 4 + 1):
        [0]             robot speed
        [1]             normalised path progress (0‒1)
        [2]             signed lateral offset (+ = left)
        [3]             heading error w.r.t. path tangent
        [4 : 4+2N]      N lookahead path points in robot frame (x,y pairs)
        [4+2N : 4+2N+4] human relative (rx, ry, rvx, rvy) in robot frame
        [-1]            corridor risk flag (1.0 if human in corridor)

    Action (float32, shape=(2,)):
        [0] forward offset of local goal in robot frame  (clamped to goal_fwd_range)
        [1] lateral offset of local goal in robot frame  (clamped to goal_lat_range)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    DEFAULT_CFG: dict = dict(
        map_size=20.0,
        dt=0.1,
        max_steps=200,
        # robot
        robot_radius=0.3,
        max_v=1.0,
        max_omega=1.0,
        init_v=0.6,
        # human
        human_radius=0.3,
        human_speed_range=(0.1, 0.3),
        # safety / collision
        collision_dist=0.6,
        safety_dist=1.5,
        # corridor
        corridor_len=8.0,
        corridor_w=1.8,
        # action bounds
        goal_fwd_range=(0.0, 3.0),
        goal_lat_range=(-2.0, 2.0),
        # observation
        n_lookahead=8,
        lookahead_spacing=1.0,
        # reward
        w_collision=-100.0,
        w_safety=-5.0,
        w_deviation=-2.0,
        w_heading=-1.0,
        w_progress=10.0,
        w_speed=2.0,
        w_time=-0.5,
        w_success=20.0,
        # termination
        success_lat_thresh=0.3,
        success_hdg_thresh=0.2,
        oob_margin=2.0,
        # human appearance
        human_delay=2.0,
        # path curvature limit (κ = 1/radius; 0.3 ≈ min radius 3.3 m)
        max_path_curvature=0.3,
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

        # will be set in reset()
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

        self._fig = None
        self._ax = None

        self._recording = False
        self._frames: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Begin collecting frames for video export."""
        self._recording = True
        self._frames = []

    def stop_recording(self, path: str = "episode.mp4", fps: int = 10) -> str | None:
        """Save collected frames to video and stop recording.

        Tries mp4 (requires ffmpeg) first, falls back to gif (Pillow).
        Returns the actual path written, or None on failure.
        """
        self._recording = False
        if not self._frames:
            print("No frames to save.")
            return None

        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis("off")
        im = ax.imshow(self._frames[0])

        def _update(i):
            im.set_data(self._frames[i])
            return [im]

        ani = FuncAnimation(
            fig, _update, frames=len(self._frames),
            interval=1000 // fps, blit=True,
        )

        saved_path: str | None = None
        if path.endswith(".gif"):
            from matplotlib.animation import PillowWriter
            ani.save(path, writer=PillowWriter(fps=fps))
            saved_path = path
        else:
            try:
                from matplotlib.animation import FFMpegWriter
                ani.save(path, writer=FFMpegWriter(fps=fps))
                saved_path = path
            except Exception:
                gif_path = path.rsplit(".", 1)[0] + ".gif"
                try:
                    from matplotlib.animation import PillowWriter
                    ani.save(gif_path, writer=PillowWriter(fps=fps))
                    saved_path = gif_path
                    print(f"ffmpeg unavailable, saved as {saved_path}")
                except Exception as e:
                    print(f"Failed to save video: {e}")

        n_frames = len(self._frames)
        plt.close(fig)
        self._frames = []

        if saved_path:
            print(f"Video saved → {saved_path}  ({n_frames} frames)")
        return saved_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_path(self, rng: np.random.Generator) -> None:
        """Generate a smooth path with bounded curvature.

        Uses densely-sampled sinusoidal combinations. After building the
        spline, checks actual curvature and scales down amplitude if needed.
        """
        ms, mg = self.cfg["map_size"], 3.0
        kmax = self.cfg["max_path_curvature"]
        lead_len = 6.0

        x_start, x_end = mg - 1.0, ms - mg
        y_mid = ms / 2.0

        n_comp = int(rng.integers(2, 4))
        amps = rng.uniform(0.3, 2.0, n_comp)
        freqs = rng.uniform(0.15, 0.5, n_comp)
        phases = rng.uniform(0, 2 * np.pi, n_comp)
        y_base = rng.uniform(y_mid - 2.0, y_mid + 2.0)

        n_pts = 80
        xs = np.linspace(x_start, x_end, n_pts)

        # Smooth ramp: flat lead-in → gradual onset of curvature
        ramp = np.clip((xs - x_start - lead_len * 0.5) / (lead_len * 0.5), 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep

        def _build(scale: float = 1.0) -> np.ndarray:
            ys = np.full_like(xs, y_base)
            for a, f, p in zip(amps, freqs, phases):
                ys += ramp * scale * a * np.sin(f * (xs - x_start) + p)
            return ys

        # Binary search for the largest scale that respects kmax
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            path = ReferencePath(np.column_stack([xs, _build(mid)]))
            if path.max_abs_curvature() <= kmax:
                lo = mid
            else:
                hi = mid

        self.path = ReferencePath(np.column_stack([xs, _build(lo)]))

    def _spawn_human(self, rng: np.random.Generator) -> None:
        """Spawn a human on a **collision course** with the robot.

        The key idea: pick an encounter point on the path ahead, compute
        the time for the robot to reach it, then place the human so that
        it arrives at the same point at (approximately) the same time.
        """
        c = self.cfg
        behav = str(rng.choice(["cross", "side", "along"]))
        speed = float(rng.uniform(*c["human_speed_range"]))

        # Average robot speed (blend of pre-sim init_v and episode max_v)
        v_r = (c["init_v"] + c["max_v"]) * 0.5

        # Time horizon until encounter (seconds, includes pre-sim time)
        t_enc = float(rng.uniform(3.0, 6.0))

        # Encounter point on the reference path
        enc_s = self.cur_s + v_r * t_enc
        enc_s = min(enc_s, self.path.total_length - 2.0)
        enc_pos = self.path.position(enc_s)
        nrm = self.path.normal(enc_s)
        tan = self.path.tangent(enc_s)
        side = rng.choice([-1.0, 1.0])

        # Slight timing jitter so it's not always a perfect head-on
        t_h = t_enc * float(rng.uniform(0.85, 1.1))

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

        else:  # along — slow human on the path, robot catches up
            slow = speed * 0.4
            # Place human so the robot closes the gap in t_enc seconds
            gap = (v_r - slow) * t_enc
            h_s0 = self.cur_s + gap
            h_s0 = float(np.clip(h_s0, self.cur_s + 1.0,
                                  self.path.total_length - 1.0))
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
        """Spawn the human on a collision course from the robot's current position."""
        self._spawn_human(self.np_random)
        self._human_visible = True

    def _update_progress(self) -> None:
        s_new, _, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
        )
        self.cur_s = max(self.cur_s, s_new)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

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

        return np.array(
            [self.rv, progress, lat, h_err] + look + [hrx, hry, hrvx, hrvy, risk],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reward(self, old_s: float, collision: bool, success: bool) -> float:
        c = self.cfg
        if collision:
            return float(c["w_collision"])

        r = 0.0
        if self._human_visible:
            dh = np.hypot(self.rx - self.hx, self.ry - self.hy)
            if dh < c["safety_dist"]:
                r += c["w_safety"] * (c["safety_dist"] - dh) / c["safety_dist"]

        _, lat, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
        )
        r += c["w_deviation"] * lat ** 2
        r += c["w_heading"] * abs(wrap_angle(self.rtheta - self.path.heading(self.cur_s)))
        r += c["w_progress"] * max(0.0, self.cur_s - old_s)
        r += c["w_speed"] * (self.rv / c["max_v"])
        r += c["w_time"]

        if success:
            r += c["w_success"]
        return float(r)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_done(self) -> tuple[bool, bool, bool, bool, dict]:
        """Returns (terminated, truncated, collision, success, info)."""
        c = self.cfg
        info: dict = {}
        terminated = truncated = collision = success = False

        if self._human_visible:
            dh = np.hypot(self.rx - self.hx, self.ry - self.hy)
            if dh < c["collision_dist"]:
                terminated, collision = True, True
                info["collision"] = True
                return terminated, truncated, collision, success, info

            _, _, lat_abs = self.path.closest_point(
                self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
            )
            h_err = abs(wrap_angle(self.rtheta - self.path.heading(self.cur_s)))
            on_path = lat_abs < c["success_lat_thresh"] and h_err < c["success_hdg_thresh"]
            cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
            h_ahead = cr * (self.hx - self.rx) + sr * (self.hy - self.ry)
            human_behind = h_ahead < 0
            human_far = dh > c["safety_dist"] * 2
            human_clear = (not self._in_corridor(self.hx, self.hy)) and (human_far or human_behind)
        else:
            on_path = human_clear = False

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

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        self._make_path(rng)
        # Start within the horizontal lead-in (first ~4 m of arc)
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

        # Spawn human at the designated step
        if not self._human_visible and self.steps >= self._human_appear_step:
            self._activate_human()

        action = np.asarray(action, dtype=np.float32)
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

        self._rtraj.append(np.array([self.rx, self.ry]))

        self._update_progress()
        terminated, truncated, collision, success, info = self._check_done()
        reward = self._reward(old_s, collision, success)
        obs = self._obs()

        info["step"] = self.steps
        if self._human_visible:
            info["behavior"] = self._h_behav
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode is None and not self._recording:
            return None
        if self._fig is None:
            if self.render_mode == "human":
                plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(10, 8))

        ax = self._ax
        ax.clear()
        c = self.cfg
        ms = c["map_size"]
        ax.set_xlim(-1, ms + 1)
        ax.set_ylim(-1, ms + 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

        px, py = self.path.get_all_xy()
        ax.plot(px, py, "b--", lw=1.5, alpha=0.4, label="Reference path")

        # corridor polygon
        s_lo = self.cur_s
        s_hi = min(self.cur_s + c["corridor_len"], self.path.total_length)
        s_arr = np.linspace(s_lo, s_hi, 40)
        left, right = [], []
        for sv in s_arr:
            p = self.path.position(sv)
            n = self.path.normal(sv)
            left.append(p + c["corridor_w"] * n)
            right.append(p - c["corridor_w"] * n)
        poly = np.array(left + right[::-1])
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.12, color="orange", label="Corridor")

        if len(self._rtraj) > 1:
            rt = np.array(self._rtraj)
            ax.plot(rt[:, 0], rt[:, 1], "g-", lw=2, alpha=0.7, label="Robot traj")
        if self._human_visible and len(self._htraj) > 1:
            ht = np.array(self._htraj)
            ax.plot(ht[:, 0], ht[:, 1], "r-", lw=1.5, alpha=0.5, label="Human traj")

        ax.add_patch(plt.Circle((self.rx, self.ry), c["robot_radius"],
                                color="green", alpha=0.7))
        al = 0.5
        ax.arrow(self.rx, self.ry,
                 al * np.cos(self.rtheta), al * np.sin(self.rtheta),
                 head_width=0.15, head_length=0.1, fc="darkgreen", ec="darkgreen")

        ax.add_patch(plt.Circle((self.rx, self.ry), c["safety_dist"],
                                fill=False, ls="--", color="gold", alpha=0.4))

        if self._human_visible:
            ax.add_patch(plt.Circle((self.hx, self.hy), c["human_radius"],
                                    color="red", alpha=0.7))
            ax.arrow(self.hx, self.hy, self.hvx * 0.8, self.hvy * 0.8,
                     head_width=0.1, head_length=0.08, fc="darkred", ec="darkred")

        if self._goals:
            g = self._goals[-1]
            ax.plot(g[0], g[1], "mx", ms=12, mew=3, label="Local goal")

        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Step {self.steps}  |  v={self.rv:.2f} m/s  |  {self._h_behav}")

        if self.render_mode == "human":
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            plt.pause(0.01)

        if self._recording or self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = buf.reshape((h, w, 3))
            if self._recording:
                self._frames.append(frame.copy())
            if self.render_mode == "rgb_array":
                return frame

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = self._ax = None


# ======================================================================
# Demo utilities
# ======================================================================

_RESULT_COLORS = {
    "COLLISION": "#d32f2f",
    "SUCCESS": "#388e3c",
    "TIMEOUT": "#f57c00",
}


def _result_tag(info: dict) -> str:
    if info.get("collision"):
        return "COLLISION"
    if info.get("success"):
        return "SUCCESS"
    if info.get("timeout"):
        return "TIMEOUT"
    if info.get("out_of_bounds"):
        return "OUT OF BOUNDS"
    if info.get("path_end"):
        return "PATH END"
    return "OTHER"


def _show_result(env, tag: str, ret: float, steps: int, wait: bool = True):
    """Render the final frame with a result banner overlaid on the plot.

    If *wait* is True, block until the user closes the window or presses a key.
    """
    if env.render_mode is None and not env._recording:
        return
    env.render()

    ax = env._ax
    color = _RESULT_COLORS.get(tag, "#555555")

    ax.text(
        0.5, 0.92, tag,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=28, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", fc=color, alpha=0.85),
    )
    ax.text(
        0.5, 0.82,
        f"steps = {steps}    return = {ret:.1f}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=14, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
    )

    if wait:
        ax.text(
            0.5, 0.02, "press any key or close window to continue",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="#888888",
        )

    env._fig.canvas.draw_idle()
    env._fig.canvas.flush_events()

    # Capture the result frame for video (hold ~1 s worth of frames)
    if env._recording:
        env._fig.canvas.draw()
        w, h = env._fig.canvas.get_width_height()
        buf = np.frombuffer(env._fig.canvas.tostring_rgb(), dtype=np.uint8)
        result_frame = buf.reshape((h, w, 3)).copy()
        for _ in range(10):
            env._frames.append(result_frame)

    if wait and env.render_mode is not None:
        plt.ioff()
        try:
            plt.waitforbuttonpress()
        except Exception:
            pass
        plt.ion()


def _obs_to_path_goal(obs, cfg, lookahead_idx: int = 2):
    """Extract the i-th lookahead path point from obs as an action.

    Returns the raw robot-frame (lx, ly) of the path point.  step() does
    NOT clip the action, so the goal lands exactly on the reference path
    regardless of curvature.
    """
    base = 4 + (lookahead_idx - 1) * 2
    return np.array([obs[base], obs[base + 1]], dtype=np.float32)


def demo_random(episodes: int = 3, render: bool = True, save_video: str | None = None):
    """Sanity check with a forward-biased random policy."""
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        print(f"\n=== Episode {ep + 1} | {info['behavior']} ===")
        print(f"    obs dim={obs.shape[0]}, action space={env.action_space}")

        ret = 0.0
        done = False
        while not done:
            a = env.action_space.sample()
            a[0] = np.clip(a[0] + 1.0, *env.cfg["goal_fwd_range"])
            a[1] *= 0.3
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc
            env.render()

        tag = _result_tag(info)
        print(f"    steps={info['step']}, return={ret:.1f}, result={tag}")
        _show_result(env, tag, ret, info["step"], wait=render)

    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_follow_path(render: bool = True, save_video: str | None = None):
    """Baseline: local goal = reference-path point 2 m ahead (no avoidance)."""
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()
    obs, info = env.reset(seed=0)
    print(f"Follow-path baseline | {info['behavior']}")

    ret, done = 0.0, False
    while not done:
        action = _obs_to_path_goal(obs, env.cfg, lookahead_idx=2)
        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = _result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    _show_result(env, tag, ret, info["step"], wait=render)
    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_reactive_avoid(render: bool = True, save_video: str | None = None):
    """Simple reactive policy: follow path but steer away when human is close."""
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()
    obs, info = env.reset(seed=0)
    print(f"Reactive avoidance | {info['behavior']}")

    cfg = env.cfg
    ret, done = 0.0, False
    while not done:
        action = _obs_to_path_goal(obs, cfg, lookahead_idx=2)

        n_lk = cfg["n_lookahead"]
        h_rx = obs[4 + 2 * n_lk]
        h_ry = obs[4 + 2 * n_lk + 1]
        dist = np.hypot(h_rx, h_ry)

        if dist < cfg["safety_dist"] * 2.0 and h_rx > 0:
            steer = -np.sign(h_ry + 1e-3) * 1.5
            action[1] = np.clip(action[1] + steer, *cfg["goal_lat_range"])
            action[0] = np.clip(action[0] * 0.6, *cfg["goal_fwd_range"])

        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = _result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    _show_result(env, tag, ret, info["step"], wait=render)
    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_dodge_behind(render: bool = True, save_video: str | None = None):
    """Dodge-behind policy: detour opposite to the human's walking direction.

    When the human approaches, the robot shifts laterally in the direction
    *opposite* to the human's velocity, effectively passing behind the
    human rather than cutting in front.
    """
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()
    obs, info = env.reset(seed=0)
    print(f"Dodge behind human | {info['behavior']}")

    cfg = env.cfg
    n_lk = cfg["n_lookahead"]
    detect_radius = cfg["safety_dist"] * 2.5
    dodge_strength = 1.5

    ret, done = 0.0, False
    while not done:
        h_rx = obs[4 + 2 * n_lk]
        h_ry = obs[4 + 2 * n_lk + 1]
        h_dist = np.hypot(h_rx, h_ry)

        human_ahead = h_rx > 0
        human_close = h_dist < detect_radius

        if human_ahead and human_close:
            action = _obs_to_path_goal(obs, cfg, lookahead_idx=2)
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
            action = _obs_to_path_goal(obs, cfg, lookahead_idx=4)
            lat_offset = obs[2]
            action[1] = np.clip(action[1] - 1.5 * lat_offset, *cfg["goal_lat_range"])

        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = _result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    _show_result(env, tag, ret, info["step"], wait=render)
    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_stop_and_wait(render: bool = True, save_video: str | None = None):
    """Stop-and-wait policy: follow path, freeze when human blocks ahead."""
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()
    obs, info = env.reset(seed=0)
    print(f"Stop-and-wait | {info['behavior']}")

    cfg = env.cfg
    n_lk = cfg["n_lookahead"]
    detect_radius = cfg["safety_dist"] * 2.5
    lane_half_w = cfg["corridor_w"] * 1.5

    ret, done = 0.0, False
    while not done:
        h_rx = obs[4 + 2 * n_lk]
        h_ry = obs[4 + 2 * n_lk + 1]
        h_dist = np.hypot(h_rx, h_ry)

        human_ahead = h_rx > 0
        human_close = h_dist < detect_radius
        human_in_lane = abs(h_ry) < lane_half_w

        if human_ahead and human_close and human_in_lane:
            action = np.array([0.0, 0.0], dtype=np.float32)
        else:
            action = _obs_to_path_goal(obs, cfg, lookahead_idx=2)

        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = _result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    _show_result(env, tag, ret, info["step"], wait=render)
    if save_video:
        env.stop_recording(save_video)
    env.close()


def demo_slow_on_path(render: bool = True, save_video: str | None = None):
    """Slow-down policy: stay on the reference path but reduce speed when human is near."""
    env = LocalPlannerEnv(render_mode="human" if render else None)
    if save_video:
        env.start_recording()
    obs, info = env.reset(seed=0)
    print(f"Slow on path | {info['behavior']}")

    cfg = env.cfg
    n_lk = cfg["n_lookahead"]
    detect_radius = cfg["safety_dist"] * 2.5
    lane_half_w = cfg["corridor_w"] * 1.5
    min_speed_ratio = 0.15

    ret, done = 0.0, False
    while not done:
        h_rx = obs[4 + 2 * n_lk]
        h_ry = obs[4 + 2 * n_lk + 1]
        h_dist = np.hypot(h_rx, h_ry)

        human_ahead = h_rx > 0
        human_close = h_dist < detect_radius
        human_in_lane = abs(h_ry) < lane_half_w

        action = _obs_to_path_goal(obs, cfg, lookahead_idx=2)

        if human_ahead and human_close and human_in_lane:
            speed_ratio = min_speed_ratio + (1.0 - min_speed_ratio) * (h_dist / detect_radius)
            action[0] = np.clip(action[0] * speed_ratio, *cfg["goal_fwd_range"])

        obs, r, term, trunc, info = env.step(action)
        ret += r
        done = term or trunc
        env.render()

    tag = _result_tag(info)
    print(f"steps={info['step']}, return={ret:.1f}, result={tag}")
    _show_result(env, tag, ret, info["step"], wait=render)
    if save_video:
        env.stop_recording(save_video)
    env.close()


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser(description="Local path planning simulator demo")
    pa.add_argument("--mode", choices=["random", "follow", "avoid", "dodge", "slow", "stop"],
                    default="follow")
    pa.add_argument("--episodes", type=int, default=3)
    pa.add_argument("--no-render", action="store_true")
    pa.add_argument("--save-video", type=str, default=None, metavar="PATH",
                    help="Save episode video (e.g. out.mp4 or out.gif)")
    args = pa.parse_args()

    _render = not args.no_render
    _vid = args.save_video

    if args.mode == "random":
        demo_random(args.episodes, _render, _vid)
    elif args.mode == "avoid":
        demo_reactive_avoid(_render, _vid)
    elif args.mode == "dodge":
        demo_dodge_behind(_render, _vid)
    elif args.mode == "slow":
        demo_slow_on_path(_render, _vid)
    elif args.mode == "stop":
        demo_stop_and_wait(_render, _vid)
    else:
        demo_follow_path(_render, _vid)
