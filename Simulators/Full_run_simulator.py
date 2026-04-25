"""
Full Run Simulator for RL-based Local Path Planning

Continuous simulation where a robot follows a long straight reference path
from start to goal. Human encounters occur at predefined points along the
path. Each encounter triggers an avoidance episode that ends on collision or
successful return to the reference path, after which the robot continues
toward the goal.

Usage:
    from Full_run_simulator import FullRunEnv, run_full_demo

    # Path-following demo (no RL model):
    run_full_demo(render=True, save_video="full_run.gif")

    # With a trained RL model:
    from stable_baselines3 import SAC
    model = SAC.load("model.zip")
    run_full_demo(rl_model=model, render=True, save_video="full_run.gif")
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from light_weight_simulator import (
    wrap_angle,
    ReferencePath,
    PurePursuitController,
    _obs_normalization_scales,
    _obs_to_path_goal,
    HybridPolicy,
)


# ======================================================================
# Environment
# ======================================================================


class FullRunEnv(gym.Env):
    """Gymnasium environment for a continuous start-to-goal run with encounters.

    The robot follows a long straight reference path. Predefined human
    encounters trigger avoidance episodes along the way. Each episode ends
    on collision or successful return to path, then the robot continues.
    The environment terminates only when the robot reaches the goal.

    Observation and action formats match LocalPlannerEnv for model compatibility.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    DEFAULT_CFG: dict = dict(
        # Straight path
        path_length=40.0,
        path_y=5.0,
        # Robot
        robot_radius=0.3,
        max_v=1.0,
        max_omega=1.0,
        dt=0.1,
        init_v=0.6,
        max_steps=1200,
        # Human
        human_radius=0.3,
        human_speed_range=(0.1, 0.3),
        # Safety / collision
        collision_dist=0.6,
        safety_dist=1.5,
        # Corridor (detection zone ahead of robot)
        corridor_len=8.0,
        corridor_w=1.8,
        # Action bounds (match LocalPlannerEnv)
        goal_fwd_range=(0.0, 3.0),
        goal_lat_range=(-2.0, 2.0),
        # Observation (match LocalPlannerEnv)
        n_lookahead=8,
        lookahead_spacing=1.0,
        normalize_obs=True,
        # Termination / success
        success_lat_thresh=0.3,
        success_hdg_thresh=0.2,
        oob_margin=5.0,
        # Encounter definitions: list of dicts
        #   s            – arc-length position on path
        #   behavior     – "cross" | "side" | "along"
        #   speed        – human walking speed (m/s)
        #   side         – 1.0 (from above/left) or -1.0 (from below/right)
        #   start_dist   – initial perpendicular distance from path (optional)
        encounters=[
            dict(s=12.0, behavior="cross", speed=0.2, side=1.0),
            dict(s=28.0, behavior="cross", speed=0.25, side=-1.0),
        ],
        # Max angular deviation from perpendicular (radians, ~25 deg)
        cross_angle_jitter=0.45,
        human_despawn_delay=15,
        # Hallway rendering
        hallway_half_width=3.0,
        # Viewport for rendering
        view_ahead=10.0,
        view_behind=6.0,
        view_half_height=6.0,
        # Kept for obs normalization compatibility with trained LocalPlannerEnv models
        map_size=20.0,
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

        self._human_visible = False
        self._encounter_idx = 0
        self._encounter_active = False
        self._encounter_resolved = False
        self._encounter_results: list[dict] = []
        self._despawn_counter = 0
        self._ep_min_d_human = float("inf")
        self._h_behav = ""
        self._prev_abs_lat = 0.0

        self._rtraj: list[np.ndarray] = []
        self._htraj: list[np.ndarray] = []
        self._goals: list[np.ndarray] = []

        # Overlay text for encounter results in video
        self._overlay_text = ""
        self._overlay_color = "#333333"
        self._overlay_ttl = 0

        self._fig = None
        self._ax = None
        self._recording = False
        self._frames: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        self._recording = True
        self._frames = []

    def stop_recording(self, path: str = "full_run.mp4", fps: int = 10) -> str | None:
        self._recording = False
        if not self._frames:
            print("No frames to save.")
            return None

        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(14, 6))
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
            print(f"Video saved -> {saved_path}  ({n_frames} frames)")
        return saved_path

    # ------------------------------------------------------------------
    # Path creation
    # ------------------------------------------------------------------

    def _make_straight_path(self) -> None:
        c = self.cfg
        length = c["path_length"]
        y = c["path_y"]
        n_pts = max(10, int(length / 2))
        xs = np.linspace(0, length, n_pts)
        ys = np.full_like(xs, y)
        self.path = ReferencePath(
            np.column_stack([xs, ys]),
            num_samples=max(1000, int(length * 20)),
        )

    # ------------------------------------------------------------------
    # Human encounter spawning
    # ------------------------------------------------------------------

    def _spawn_encounter_human(self, enc: dict) -> None:
        s_enc = enc["s"]
        speed = enc.get("speed", 0.2)
        side = enc.get("side", 1.0)
        start_dist = enc.get("start_dist", 4.0)

        enc_pos = self.path.position(s_enc)
        nrm = self.path.normal(s_enc)

        # Base direction: perpendicular, then add random angular jitter
        jitter_max = self.cfg.get("cross_angle_jitter", 0.45)
        ang = float(self.np_random.uniform(-jitter_max, jitter_max))
        ca, sa = np.cos(ang), np.sin(ang)
        d_base = -side * nrm
        d = np.array([d_base[0] * ca - d_base[1] * sa,
                       d_base[0] * sa + d_base[1] * ca])

        start = enc_pos - d * start_dist
        self.hvx, self.hvy = float(d[0] * speed), float(d[1] * speed)

        self.hx, self.hy = float(start[0]), float(start[1])
        self._h_behav = enc.get("behavior", "cross")
        self._human_visible = True
        self._encounter_active = False
        self._encounter_resolved = False
        self._ep_min_d_human = float("inf")

    def _maybe_spawn_next_human(self) -> None:
        c = self.cfg
        encounters = c.get("encounters", [])
        if self._human_visible or self._encounter_idx >= len(encounters):
            return

        enc = encounters[self._encounter_idx]
        speed = enc.get("speed", 0.2)
        start_dist = enc.get("start_dist", 4.0)

        v_est = max(self.rv, c["init_v"]) * 0.9
        t_human = start_dist / max(speed, 0.05)
        auto_appear = v_est * t_human
        appear_dist = enc.get("appear_distance", max(auto_appear, 6.0))

        if self.cur_s >= enc["s"] - appear_dist:
            self._spawn_encounter_human(enc)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

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

    def _update_progress(self) -> None:
        s_new, _, _ = self.path.closest_point(
            self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
        )
        self.cur_s = max(self.cur_s, s_new)

    # ------------------------------------------------------------------
    # Observation (same format as LocalPlannerEnv)
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

        vec = [self.rv, progress, lat, h_err] + look + [hrx, hry, hrvx, hrvy, risk]
        if c.get("normalize_obs", False):
            lat_s, pos_s, vel_s, ms, mv = _obs_normalization_scales(c)
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

    # ------------------------------------------------------------------
    # Encounter management
    # ------------------------------------------------------------------

    def _resolve_encounter(self, result: str) -> dict:
        self._encounter_active = False
        self._encounter_resolved = True
        self._despawn_counter = self.cfg.get("human_despawn_delay", 15)
        rec = {
            "idx": self._encounter_idx,
            "result": result,
            "min_dist": float(self._ep_min_d_human),
        }
        self._encounter_results.append(rec)

        if result == "collision":
            self._overlay_text = "COLLISION!"
            self._overlay_color = "#d32f2f"
            self._overlay_ttl = 20
        elif result == "success":
            self._overlay_text = "AVOIDED"
            self._overlay_color = "#388e3c"
            self._overlay_ttl = 15

        return rec

    def _check_encounter_events(self) -> dict:
        c = self.cfg
        info: dict = {}

        if not self._human_visible:
            return info

        # Handle despawn countdown after encounter resolved
        if self._encounter_resolved:
            self._despawn_counter -= 1
            if self._despawn_counter <= 0:
                self._human_visible = False
                self._encounter_idx += 1
                info["encounter_despawned"] = True
            return info

        dh = float(np.hypot(self.rx - self.hx, self.ry - self.hy))

        # 1. Collision check (always)
        if dh < c["collision_dist"]:
            rec = self._resolve_encounter("collision")
            info["encounter_collision"] = True
            info["encounter_result"] = rec
            return info

        # 2. Encounter activation: human enters corridor
        if not self._encounter_active:
            if self._in_corridor(self.hx, self.hy):
                self._encounter_active = True
                info["encounter_start"] = True

        # 3. Success check: robot back on path + human clear
        if self._encounter_active:
            _, lat, _ = self.path.closest_point(
                self.rx, self.ry, s_hint=self.cur_s, search_radius=5.0,
            )
            h_err = abs(wrap_angle(self.rtheta - self.path.heading(self.cur_s)))
            on_path = abs(lat) < c["success_lat_thresh"] and h_err < c["success_hdg_thresh"]

            cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
            h_ahead = cr * (self.hx - self.rx) + sr * (self.hy - self.ry)
            human_behind = h_ahead < 0
            human_far = dh > c["safety_dist"] * 2
            human_clear = (not self._in_corridor(self.hx, self.hy)) and (
                human_far or human_behind
            )

            if on_path and human_clear:
                rec = self._resolve_encounter("success")
                info["encounter_success"] = True
                info["encounter_result"] = rec

        # 4. Skip: human passed without ever entering corridor
        if not self._encounter_active and not self._encounter_resolved:
            cr, sr = np.cos(self.rtheta), np.sin(self.rtheta)
            h_ahead_dist = cr * (self.hx - self.rx) + sr * (self.hy - self.ry)
            if h_ahead_dist < -5.0 or dh > 15.0:
                self._human_visible = False
                self._encounter_idx += 1
                self._encounter_results.append({
                    "idx": self._encounter_idx - 1,
                    "result": "no_conflict",
                    "min_dist": float(self._ep_min_d_human),
                })
                info["encounter_skipped"] = True

        return info

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._make_straight_path()
        self.cur_s = 0.5

        p = self.path.position(self.cur_s)
        h = self.path.heading(self.cur_s)
        self.rx, self.ry, self.rtheta = float(p[0]), float(p[1]), h
        self.rv = 0.0

        self._human_visible = False
        self._encounter_idx = 0
        self._encounter_active = False
        self._encounter_resolved = False
        self._encounter_results = []
        self._despawn_counter = 0
        self._h_behav = ""
        self.hx = self.hy = self.hvx = self.hvy = 0.0

        self.steps = 0
        self._ep_min_d_human = float("inf")
        self._prev_abs_lat = 0.0
        self._rtraj = [np.array([self.rx, self.ry])]
        self._htraj = []
        self._goals = []

        self._overlay_text = ""
        self._overlay_ttl = 0

        obs = self._obs()
        info = {"phase": "cruise"}
        return obs, info

    def step(self, action):
        c = self.cfg
        self.steps += 1

        self._maybe_spawn_next_human()

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

        enc_info = self._check_encounter_events()

        terminated = truncated = False
        info = dict(enc_info)
        info["step"] = self.steps

        if self.cur_s >= self.path.total_length - 1.0:
            terminated = True
            info["goal_reached"] = True

        if self.steps >= c["max_steps"]:
            truncated = True
            info["timeout"] = True

        margin = c.get("oob_margin", 5.0)
        path_y = c["path_y"]
        if (
            self.rx < -margin
            or self.rx > c["path_length"] + margin
            or self.ry < path_y - margin * 3
            or self.ry > path_y + margin * 3
        ):
            truncated = True
            info["out_of_bounds"] = True

        if self._encounter_active:
            info["phase"] = "encounter"
        elif self._human_visible:
            info["phase"] = "human_visible"
        else:
            info["phase"] = "cruise"

        if terminated or truncated:
            info["encounter_results"] = list(self._encounter_results)

        reward = 0.0
        obs = self._obs()
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
            self._fig, self._ax = plt.subplots(figsize=(14, 6))

        ax = self._ax
        ax.clear()
        c = self.cfg

        va = c.get("view_ahead", 10.0)
        vb = c.get("view_behind", 6.0)
        vh = c.get("view_half_height", 6.0)

        x_lo = self.rx - vb
        x_hi = self.rx + va
        y_lo = c["path_y"] - vh
        y_hi = c["path_y"] + vh

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

        # Hallway walls and floor
        hw = c.get("hallway_half_width", 3.0)
        wall_top = c["path_y"] + hw
        wall_bot = c["path_y"] - hw
        ax.fill_between(
            [x_lo, x_hi], wall_bot, wall_top,
            color="#f5f0e1", alpha=0.3, zorder=0,
        )
        ax.hlines(
            [wall_top, wall_bot], x_lo, x_hi,
            colors="#8B4513", linewidths=2.5, alpha=0.6, zorder=1,
        )

        # Reference path
        px, py = self.path.get_all_xy()
        ax.plot(px, py, "b--", lw=1.5, alpha=0.4, label="Reference path")

        # Start and goal markers
        p_start = self.path.position(0)
        p_goal = self.path.position(self.path.total_length)
        ax.plot(p_start[0], p_start[1], "s", color="green", ms=12, label="Start", zorder=5)
        ax.plot(p_goal[0], p_goal[1], "*", color="red", ms=16, label="Goal", zorder=5)

        # Corridor detection zone
        s_lo_c = self.cur_s
        s_hi_c = min(self.cur_s + c["corridor_len"], self.path.total_length)
        s_arr = np.linspace(s_lo_c, s_hi_c, 40)
        left, right = [], []
        for sv in s_arr:
            p = self.path.position(sv)
            n = self.path.normal(sv)
            left.append(p + c["corridor_w"] * n)
            right.append(p - c["corridor_w"] * n)
        poly = np.array(left + right[::-1])
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.12, color="orange", label="Corridor")

        # Robot trajectory
        if len(self._rtraj) > 1:
            rt = np.array(self._rtraj)
            ax.plot(rt[:, 0], rt[:, 1], "g-", lw=2, alpha=0.7, label="Robot traj")

        # Human trajectory
        if self._human_visible and len(self._htraj) > 1:
            ht = np.array(self._htraj)
            ax.plot(ht[:, 0], ht[:, 1], "r-", lw=1.5, alpha=0.5, label="Human traj")

        # Robot
        ax.add_patch(plt.Circle(
            (self.rx, self.ry), c["robot_radius"],
            color="green", alpha=0.7, zorder=10,
        ))
        al = 0.5
        ax.arrow(
            self.rx, self.ry,
            al * np.cos(self.rtheta), al * np.sin(self.rtheta),
            head_width=0.15, head_length=0.1, fc="darkgreen", ec="darkgreen",
            zorder=11,
        )
        ax.add_patch(plt.Circle(
            (self.rx, self.ry), c["safety_dist"],
            fill=False, ls="--", color="gold", alpha=0.4, zorder=9,
        ))

        # Human
        if self._human_visible:
            ax.add_patch(plt.Circle(
                (self.hx, self.hy), c["human_radius"],
                color="red", alpha=0.7, zorder=10,
            ))
            ax.arrow(
                self.hx, self.hy, self.hvx * 0.8, self.hvy * 0.8,
                head_width=0.1, head_length=0.08, fc="darkred", ec="darkred",
                zorder=11,
            )

        # Local goal
        if self._goals:
            g = self._goals[-1]
            ax.plot(g[0], g[1], "mx", ms=12, mew=3, label="Local goal", zorder=8)

        # Encounter position markers
        encounters = c.get("encounters", [])
        for i, enc in enumerate(encounters):
            ep = self.path.position(enc["s"])
            color = "#cccccc"
            if i < len(self._encounter_results):
                res = self._encounter_results[i]["result"]
                color = "#d32f2f" if res == "collision" else (
                    "#388e3c" if res == "success" else "#999999"
                )
            elif i == self._encounter_idx and self._encounter_active:
                color = "#ff9800"
            ax.plot(ep[0], ep[1], "D", ms=8, color=color, alpha=0.7, zorder=6)

        # Overlay text (collision / avoided banners)
        if self._overlay_ttl > 0:
            alpha = min(1.0, self._overlay_ttl / 8.0)
            ax.text(
                0.5, 0.88, self._overlay_text,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=22, fontweight="bold", color="white",
                bbox=dict(
                    boxstyle="round,pad=0.4", fc=self._overlay_color,
                    alpha=0.85 * alpha,
                ),
                zorder=20,
            )
            self._overlay_ttl -= 1

        # Title bar
        phase = "ENCOUNTER" if self._encounter_active else "CRUISE"
        if self._encounter_resolved and self._human_visible:
            phase = "RESOLVED"
        progress_pct = (self.cur_s / self.path.total_length) * 100
        n_done = len(self._encounter_results)
        n_total = len(encounters)
        title = (
            f"Step {self.steps}  |  v={self.rv:.2f} m/s  |  "
            f"{phase}  |  Progress: {progress_pct:.0f}%  |  "
            f"Encounters: {n_done}/{n_total}"
        )
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)

        if self.render_mode == "human":
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            try:
                plt.pause(0.01)
            except Exception:
                pass

        if self._recording or self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            frame = np.asarray(self._fig.canvas.buffer_rgba())[..., :3]
            if self._recording:
                self._frames.append(frame.copy())
            if self.render_mode == "rgb_array":
                return frame

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = self._ax = None


# ======================================================================
# Demo runner
# ======================================================================


def run_full_demo(
    rl_model=None,
    config: dict | None = None,
    seed: int = 42,
    render: bool = True,
    save_video: str | None = None,
    use_hybrid: bool = True,
) -> dict:
    """Run a complete start-to-goal demonstration.

    Args:
        rl_model: Trained RL model (e.g. from stable-baselines3).
                  If None, uses pure path-following.
        config: Optional config dict overrides.
        seed: Random seed.
        render: Whether to display live rendering.
        save_video: Path to save video file (e.g. "full_run.gif").
        use_hybrid: If True and rl_model is provided, use HybridPolicy.

    Returns:
        Dict with run summary (steps, goal_reached, encounter_results).
    """
    env = FullRunEnv(config=config, render_mode="human" if render else None)
    if save_video:
        env.start_recording()

    obs, info = env.reset(seed=seed)

    policy = None
    if rl_model is not None and use_hybrid:
        policy = HybridPolicy(rl_model, env, follow_lookahead=3)
        policy.reset()

    n_enc = len(env.cfg.get("encounters", []))
    print(
        f"Starting full run: path_length={env.cfg['path_length']:.0f}m, "
        f"encounters={n_enc}"
    )

    done = False
    while not done:
        if policy is not None:
            action, state = policy.predict(obs, deterministic=True)
        else:
            action = _obs_to_path_goal(obs, env.cfg, lookahead_idx=3)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()

        if "encounter_start" in info:
            idx = env._encounter_idx
            print(f"  Encounter {idx + 1} started! ({env._h_behav})")
        if "encounter_collision" in info:
            print(f"  Encounter ended: COLLISION")
        if "encounter_success" in info:
            print(f"  Encounter ended: AVOIDED")
        if "encounter_skipped" in info:
            print(f"  Encounter skipped: no conflict")

    # Final result
    if info.get("goal_reached"):
        print(f"\nGoal reached in {info['step']} steps")
    elif info.get("timeout"):
        print(f"\nTimeout at {info['step']} steps")
    elif info.get("out_of_bounds"):
        print(f"\nOut of bounds at {info['step']} steps")

    results = info.get("encounter_results", [])
    for r in results:
        status = "COLLISION" if r["result"] == "collision" else r["result"].upper()
        print(f"  Encounter {r['idx'] + 1}: {status} (min_dist={r['min_dist']:.2f}m)")

    # Final overlay on the rendering
    if render or env._recording:
        env.render()
        ax = env._ax
        if ax is not None:
            tag = "GOAL REACHED" if info.get("goal_reached") else "INCOMPLETE"
            color = "#388e3c" if info.get("goal_reached") else "#f57c00"
            ax.text(
                0.5, 0.92, tag,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=24, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4", fc=color, alpha=0.85),
                zorder=30,
            )
            n_col = sum(1 for r in results if r["result"] == "collision")
            n_suc = sum(1 for r in results if r["result"] == "success")
            summary = f"Steps: {info['step']}  |  Collisions: {n_col}  |  Avoided: {n_suc}"
            ax.text(
                0.5, 0.82, summary,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=12, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
                zorder=30,
            )
            env._fig.canvas.draw_idle()
            env._fig.canvas.flush_events()
            if env._recording:
                env._fig.canvas.draw()
                result_frame = np.asarray(
                    env._fig.canvas.buffer_rgba()
                )[..., :3].copy()
                for _ in range(15):
                    env._frames.append(result_frame)

    if save_video:
        env.stop_recording(save_video)

    env.close()
    return {
        "steps": info["step"],
        "goal_reached": info.get("goal_reached", False),
        "encounter_results": results,
    }


if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="Full run simulator demo")
    pa.add_argument("--no-render", action="store_true")
    pa.add_argument("--save-video", type=str, default=None, metavar="PATH",
                    help="Save full run video (e.g. full_run.gif)")
    pa.add_argument("--path-length", type=float, default=60.0)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--model", type=str, default=None,
                    help="Path to trained RL model (.zip)")
    pa.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac"],
                    help="RL algorithm used for the model")
    args = pa.parse_args()

    cfg = {"path_length": args.path_length}

    rl = None
    if args.model:
        from stable_baselines3 import PPO, SAC
        AlgoCls = {"ppo": PPO, "sac": SAC}[args.algo.lower()]
        rl = AlgoCls.load(args.model)

    run_full_demo(
        rl_model=rl,
        config=cfg,
        seed=args.seed,
        render=not args.no_render,
        save_video=args.save_video,
    )
