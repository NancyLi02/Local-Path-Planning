"""
Multi-Robot Simulator

Four robots follow reference paths with pure path-following (no humans).
Paths are arranged as two parallel horizontal lanes and two parallel vertical
lanes (perpendicular to the horizontal pair), forming a crossing layout.

Examples:

1. Path-following only:
    python3 Simulators/multi-robot_simulator.py

2. Save video + report:
    python3 Simulators/multi-robot_simulator.py --save-video mr_run.gif

3. Test SAC policy (nearest robot treated as dynamic obstacle in obs):
    python3 Simulators/multi-robot_simulator.py --model logs/SAC/sac_v2/sac_v2.zip --algo sac

4. Stop when another robot enters corridor:
    python3 Simulators/multi-robot_simulator.py --policy stop_corridor
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

from Simulators.Single_robot_simulator.path import (
    ReferencePath,
    obs_normalization_scales,
    wrap_angle,
)
from Simulators.Single_robot_simulator.controller import PurePursuitController
from Simulators.Single_robot_simulator.policies import obs_to_path_goal


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_video_dir() -> str:
    return os.path.join(_project_root(), "Evaluation_video", "multi_robot")


def _resolve_run_output_paths(save_video: str) -> tuple[str, str, str]:
    """Return (run_dir, video_path, report_path)."""
    base_name = os.path.basename(save_video) or "multi_robot.gif"
    run_name = os.path.splitext(base_name)[0] or "multi_robot"

    if os.path.isabs(save_video):
        parent = os.path.dirname(os.path.abspath(save_video)) or os.getcwd()
    else:
        parent = os.path.abspath(_default_video_dir())

    run_dir = os.path.join(parent, run_name)
    if os.path.exists(run_dir):
        n = 1
        while os.path.exists(f"{run_dir}_{n:03d}"):
            n += 1
        run_dir = f"{run_dir}_{n:03d}"

    video_path = os.path.join(run_dir, base_name)
    report_path = os.path.join(run_dir, "report.md")
    return run_dir, video_path, report_path


def _elapsed_seconds(steps: int, dt: float) -> float:
    return float(steps) * float(dt)


_MULTI_ROBOT_DEFAULT_CFG: dict = dict(
    n_robots=4,
    path_length=35.0,
    # Horizontal parallel pair (eastbound, y = const)
    horizontal_lanes=(8.0, 12.0),
    horizontal_start_x=2.5,
    # Vertical parallel pair (northbound, x = const) — perpendicular to horizontal
    vertical_lanes=(15.0, 19.0),
    vertical_start_y=2.5,
    # Stagger arc-length starts so robots do not all meet at once
    start_s_offsets=(0.5, 6.0, 0.5, 10.0),
    robot_radius=0.3,
    max_v=1.0,
    max_omega=1.0,
    dt=0.1,
    init_v=0.6,
    max_steps=1500,
    collision_dist=0.6,
    safety_dist=1.5,
    # Kept for obs normalization compatibility (no humans in this simulator)
    human_speed_range=(0.1, 1.0),
    corridor_len=8.0,
    corridor_w=1.8,
    rl_trigger_forward_dist=7.0,
    rl_trigger_lateral_dist=2.2,
    rl_trigger_behind_dist=0.5,
    goal_fwd_range=(0.0, 3.0),
    goal_lat_range=(-2.0, 2.0),
    n_lookahead=8,
    lookahead_spacing=1.0,
    normalize_obs=True,
    success_lat_thresh=0.05,
    success_hdg_thresh=0.1,
    oob_margin=5.0,
    map_size=40.0,
    view_margin=4.0,
    hallway_half_width=3.0,
)

ROBOT_COLORS = ("#2e7d32", "#1565c0", "#ef6c00", "#6a1b9a")
ROBOT_LABELS = ("R0 (H)", "R1 (H)", "R2 (V)", "R3 (V)")


def _make_straight_path(
    orientation: str,
    offset: float,
    length: float,
    *,
    start_coord: float,
) -> ReferencePath:
    """Build a horizontal or vertical reference path."""
    n_pts = max(10, int(length / 2))
    if orientation == "horizontal":
        xs = np.linspace(start_coord, start_coord + length, n_pts)
        ys = np.full_like(xs, offset)
    elif orientation == "vertical":
        ys = np.linspace(start_coord, start_coord + length, n_pts)
        xs = np.full_like(ys, offset)
    else:
        raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")
    return ReferencePath(
        np.column_stack([xs, ys]),
        num_samples=max(1000, int(length * 20)),
    )


def build_robot_path_specs(cfg: dict | None = None) -> list[dict]:
    """Return path specs for 4 robots: 2 parallel horizontal + 2 parallel vertical."""
    c = dict(_MULTI_ROBOT_DEFAULT_CFG)
    if cfg:
        c.update(cfg)

    h_lanes = tuple(c["horizontal_lanes"])
    v_lanes = tuple(c["vertical_lanes"])
    offsets = tuple(c.get("start_s_offsets", (0.5, 6.0, 0.5, 10.0)))

    if len(h_lanes) != 2 or len(v_lanes) != 2:
        raise ValueError("horizontal_lanes and vertical_lanes must each have length 2")

    length = float(c["path_length"])
    specs = [
        {
            "id": 0,
            "orientation": "horizontal",
            "offset": float(h_lanes[0]),
            "length": length,
            "start_coord": float(c["horizontal_start_x"]),
            "start_s": float(offsets[0]),
            "label": "horizontal lane 1",
        },
        {
            "id": 1,
            "orientation": "horizontal",
            "offset": float(h_lanes[1]),
            "length": length,
            "start_coord": float(c["horizontal_start_x"]),
            "start_s": float(offsets[1]),
            "label": "horizontal lane 2",
        },
        {
            "id": 2,
            "orientation": "vertical",
            "offset": float(v_lanes[0]),
            "length": length,
            "start_coord": float(c["vertical_start_y"]),
            "start_s": float(offsets[2]),
            "label": "vertical lane 1",
        },
        {
            "id": 3,
            "orientation": "vertical",
            "offset": float(v_lanes[1]),
            "length": length,
            "start_coord": float(c["vertical_start_y"]),
            "start_s": float(offsets[3]),
            "label": "vertical lane 2",
        },
    ]
    return specs


def _format_layout_summary(specs: list[dict]) -> str:
    lines = [
        "Layout: 2 parallel horizontal paths + 2 parallel vertical paths (no humans)",
    ]
    for spec in specs:
        ori = spec["orientation"][0].upper()
        if spec["orientation"] == "horizontal":
            pos = f"y={spec['offset']:.1f}m"
        else:
            pos = f"x={spec['offset']:.1f}m"
        lines.append(
            f"  Robot {spec['id']}: {ori}  {pos}  "
            f"length={spec['length']:.1f}m  start_s={spec['start_s']:.1f}m"
        )
    return "\n".join(lines)


def _write_run_report(
    report_path: str,
    *,
    steps: int,
    dt: float,
    all_goals_reached: bool,
    collision_pairs: list[tuple[int, int]],
    min_pairwise_dist: float,
    robot_progress: list[float],
    policy_mode: str,
    seed: int,
    video_path: str | None = None,
) -> str:
    elapsed_s = _elapsed_seconds(steps, dt)
    status = "All goals reached" if all_goals_reached else "Incomplete"

    lines = [
        "# Multi-Robot Run Report",
        "",
        "## Summary",
        "",
        f"- **Status:** {status}",
        f"- **Total time elapsed:** {elapsed_s:.1f} s",
        f"- **Steps:** {steps}",
        f"- **Robots:** 4 (2 horizontal parallel + 2 vertical parallel)",
        f"- **Policy:** {policy_mode}",
        f"- **Seed:** {seed}",
        f"- **Collisions:** {len(collision_pairs)}",
        f"- **Min pairwise distance:** "
        f"{min_pairwise_dist:.2f} m"
        if min_pairwise_dist != float("inf")
        else f"- **Min pairwise distance:** N/A",
        "",
    ]
    if video_path:
        lines.extend([
            f"- **Video:** `{os.path.basename(video_path)}`",
            "",
        ])

    lines.extend([
        "## Robot Progress",
        "",
        "| Robot | Orientation | Progress |",
        "|-------|-------------|----------|",
    ])
    orientations = ("Horizontal", "Horizontal", "Vertical", "Vertical")
    for i, prog in enumerate(robot_progress):
        lines.append(f"| {i} ({ROBOT_LABELS[i]}) | {orientations[i]} | {prog * 100:.0f}% |")

    if collision_pairs:
        lines.extend([
            "",
            "## Collision Pairs",
            "",
        ])
        for a, b in collision_pairs:
            lines.append(f"- Robot {a} & Robot {b}")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    return report_path


# ======================================================================
# Environment
# ======================================================================


class MultiRobotEnv(gym.Env):
    """Gymnasium environment for four robots on crossing reference paths.

    Two robots share parallel horizontal lanes; two share parallel vertical
    lanes (perpendicular to the horizontal pair). No humans are present.
    Other robots appear in each agent's observation in the human slots.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    DEFAULT_CFG: dict = _MULTI_ROBOT_DEFAULT_CFG

    def __init__(self, config: dict | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = dict(self.DEFAULT_CFG)
        if config:
            self.cfg.update(config)

        self.n_robots = int(self.cfg["n_robots"])
        self.render_mode = render_mode

        c = self.cfg
        n_lk = c["n_lookahead"]
        self._obs_dim = 1 + 3 + 2 * n_lk + 4 + 1

        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(self.n_robots * self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array(
                [c["goal_fwd_range"][0], c["goal_lat_range"][0]] * self.n_robots,
                dtype=np.float32,
            ),
            high=np.array(
                [c["goal_fwd_range"][1], c["goal_lat_range"][1]] * self.n_robots,
                dtype=np.float32,
            ),
        )

        self.controller = PurePursuitController(c["max_v"], c["max_omega"])
        self.path_specs: list[dict] = []
        self.paths: list[ReferencePath] = []

        self.rx = np.zeros(self.n_robots)
        self.ry = np.zeros(self.n_robots)
        self.rtheta = np.zeros(self.n_robots)
        self.rv = np.zeros(self.n_robots)
        self.cur_s = np.zeros(self.n_robots)

        self.steps = 0
        self._rtraj: list[list[np.ndarray]] = [[] for _ in range(self.n_robots)]
        self._goals: list[list[np.ndarray]] = [[] for _ in range(self.n_robots)]
        self._collision_pairs: list[tuple[int, int]] = []
        self._min_pairwise_dist = float("inf")

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

    def stop_recording(self, path: str = "multi_robot.gif", fps: int = 10) -> str | None:
        self._recording = False
        if not self._frames:
            print("No frames to save.")
            return None

        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(10, 10))
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
    # Path / geometry
    # ------------------------------------------------------------------

    def _build_paths(self) -> None:
        self.path_specs = build_robot_path_specs(self.cfg)
        self.paths = [
            _make_straight_path(
                spec["orientation"],
                spec["offset"],
                spec["length"],
                start_coord=spec["start_coord"],
            )
            for spec in self.path_specs
        ]

    def _in_corridor(self, robot_idx: int, px: float, py: float) -> bool:
        c = self.cfg
        path = self.paths[robot_idx]
        s_lo = self.cur_s[robot_idx]
        s_hi = min(s_lo + c["corridor_len"], path.total_length)
        mid = (s_lo + s_hi) / 2
        rad = (s_hi - s_lo) / 2 + 1.0
        s_cl, _, dist = path.closest_point(px, py, s_hint=mid, search_radius=rad)
        if s_cl < s_lo - 0.3 or s_cl > s_hi + 0.3:
            return False
        return dist < c["corridor_w"]

    def _nearest_other_robot(self, idx: int) -> tuple[int, float] | tuple[None, float]:
        best_j: int | None = None
        best_d = float("inf")
        for j in range(self.n_robots):
            if j == idx:
                continue
            d = float(np.hypot(self.rx[idx] - self.rx[j], self.ry[idx] - self.ry[j]))
            if d < best_d:
                best_d = d
                best_j = j
        return best_j, best_d

    def _obs_for_robot(self, idx: int) -> np.ndarray:
        c = self.cfg
        path = self.paths[idx]
        s = self.cur_s[idx]
        _, lat, _ = path.closest_point(
            self.rx[idx], self.ry[idx], s_hint=s, search_radius=5.0,
        )
        h_err = wrap_angle(self.rtheta[idx] - path.heading(s))
        progress = s / path.total_length

        cr, sr = np.cos(self.rtheta[idx]), np.sin(self.rtheta[idx])

        look: list[float] = []
        for i in range(1, c["n_lookahead"] + 1):
            sa = min(s + i * c["lookahead_spacing"], path.total_length)
            p = path.position(sa)
            dx, dy = p[0] - self.rx[idx], p[1] - self.ry[idx]
            look.extend([cr * dx + sr * dy, -sr * dx + cr * dy])

        other_idx, _ = self._nearest_other_robot(idx)
        if other_idx is not None:
            ox, oy = self.rx[other_idx], self.ry[other_idx]
            ovx = self.rv[other_idx] * np.cos(self.rtheta[other_idx])
            ovy = self.rv[other_idx] * np.sin(self.rtheta[other_idx])
            dx, dy = ox - self.rx[idx], oy - self.ry[idx]
            hrx = cr * dx + sr * dy
            hry = -sr * dx + cr * dy
            dvx = ovx - self.rv[idx] * np.cos(self.rtheta[idx])
            dvy = ovy - self.rv[idx] * np.sin(self.rtheta[idx])
            hrvx = cr * dvx + sr * dvy
            hrvy = -sr * dvx + cr * dvy
            risk = 1.0 if self._in_corridor(idx, ox, oy) else 0.0
        else:
            hrx, hry, hrvx, hrvy, risk = 10.0, 0.0, 0.0, 0.0, 0.0

        vec = [self.rv[idx], progress, lat, h_err] + look + [hrx, hry, hrvx, hrvy, risk]
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

    def _obs(self) -> np.ndarray:
        parts = [self._obs_for_robot(i) for i in range(self.n_robots)]
        return np.concatenate(parts)

    def _update_progress(self, idx: int) -> None:
        s_new, _, _ = self.paths[idx].closest_point(
            self.rx[idx], self.ry[idx],
            s_hint=self.cur_s[idx], search_radius=5.0,
        )
        self.cur_s[idx] = max(self.cur_s[idx], s_new)

    def _robot_reached_goal(self, idx: int) -> bool:
        return self.cur_s[idx] >= self.paths[idx].total_length - 1.0

    def _check_collisions(self) -> list[tuple[int, int]]:
        c = self.cfg
        pairs: list[tuple[int, int]] = []
        for i in range(self.n_robots):
            for j in range(i + 1, self.n_robots):
                d = float(np.hypot(self.rx[i] - self.rx[j], self.ry[i] - self.ry[j]))
                self._min_pairwise_dist = min(self._min_pairwise_dist, d)
                if d < c["collision_dist"]:
                    pairs.append((i, j))
        return pairs

    def _scene_bounds(self) -> tuple[float, float, float, float]:
        margin = float(self.cfg.get("view_margin", 4.0))
        xs = list(self.rx) + [spec["start_coord"] + spec["length"]
                              for spec in self.path_specs if spec["orientation"] == "horizontal"]
        ys = list(self.ry) + [spec["start_coord"] + spec["length"]
                              for spec in self.path_specs if spec["orientation"] == "vertical"]
        for spec in self.path_specs:
            if spec["orientation"] == "horizontal":
                ys.append(spec["offset"])
            else:
                xs.append(spec["offset"])
        return (
            min(xs) - margin,
            max(xs) + margin,
            min(ys) - margin,
            max(ys) + margin,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._build_paths()

        for i, spec in enumerate(self.path_specs):
            s0 = float(spec["start_s"])
            self.cur_s[i] = s0
            p = self.paths[i].position(s0)
            self.rx[i], self.ry[i] = float(p[0]), float(p[1])
            self.rtheta[i] = self.paths[i].heading(s0)
            self.rv[i] = 0.0
            self._rtraj[i] = [np.array([self.rx[i], self.ry[i]])]
            self._goals[i] = []

        self.steps = 0
        self._collision_pairs = []
        self._min_pairwise_dist = float("inf")

        obs = self._obs()
        info = {
            "path_specs": list(self.path_specs),
            "phase": "cruise",
        }
        return obs, info

    def step(self, action):
        c = self.cfg
        self.steps += 1

        action = np.asarray(action, dtype=np.float32).reshape(self.n_robots, 2)
        action = np.clip(action, self.action_space.low.reshape(self.n_robots, 2),
                         self.action_space.high.reshape(self.n_robots, 2))

        dt = c["dt"]
        for i in range(self.n_robots):
            fwd, lat = float(action[i, 0]), float(action[i, 1])
            cr, sr = np.cos(self.rtheta[i]), np.sin(self.rtheta[i])
            goal = np.array([
                self.rx[i] + fwd * cr - lat * sr,
                self.ry[i] + fwd * sr + lat * cr,
            ])
            self._goals[i].append(goal.copy())

            if abs(fwd) < 0.05 and abs(lat) < 0.05:
                v_cmd, w_cmd = 0.0, 0.0
            else:
                v_cmd, w_cmd = self.controller.compute(
                    self.rx[i], self.ry[i], self.rtheta[i], goal,
                )

            v = float(np.clip(v_cmd, 0, c["max_v"]))
            w = float(np.clip(w_cmd, -c["max_omega"], c["max_omega"]))
            self.rx[i] += v * np.cos(self.rtheta[i]) * dt
            self.ry[i] += v * np.sin(self.rtheta[i]) * dt
            self.rtheta[i] = wrap_angle(self.rtheta[i] + w * dt)
            self.rv[i] = v
            self._rtraj[i].append(np.array([self.rx[i], self.ry[i]]))
            self._update_progress(i)

        pairs = self._check_collisions()
        if pairs:
            for pair in pairs:
                if pair not in self._collision_pairs:
                    self._collision_pairs.append(pair)

        all_goals = all(self._robot_reached_goal(i) for i in range(self.n_robots))
        terminated = truncated = False
        info: dict = {"step": self.steps}

        if all_goals:
            terminated = True
            info["all_goals_reached"] = True

        if pairs:
            terminated = True
            info["robot_collision"] = True
            info["collision_pairs"] = list(pairs)

        if self.steps >= c["max_steps"]:
            truncated = True
            info["timeout"] = True

        margin = c.get("oob_margin", 5.0)
        ms = float(c.get("map_size", 40.0))
        for i in range(self.n_robots):
            if (
                self.rx[i] < -margin
                or self.rx[i] > ms + margin
                or self.ry[i] < -margin
                or self.ry[i] > ms + margin
            ):
                truncated = True
                info["out_of_bounds"] = True
                break

        info["robot_progress"] = [
            float(self.cur_s[i] / self.paths[i].total_length)
            for i in range(self.n_robots)
        ]
        info["min_pairwise_dist"] = float(self._min_pairwise_dist)

        if terminated or truncated:
            info["collision_pairs"] = list(self._collision_pairs)

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
            self._fig, self._ax = plt.subplots(figsize=(10, 10))

        ax = self._ax
        ax.clear()
        c = self.cfg

        x_lo, x_hi, y_lo, y_hi = self._scene_bounds()
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

        hw = c.get("hallway_half_width", 3.0)
        for spec in self.path_specs:
            if spec["orientation"] == "horizontal":
                y = spec["offset"]
                ax.fill_between(
                    [spec["start_coord"], spec["start_coord"] + spec["length"]],
                    y - hw, y + hw,
                    color="#f5f0e1", alpha=0.25, zorder=0,
                )
                ax.hlines(
                    [y + hw, y - hw],
                    spec["start_coord"], spec["start_coord"] + spec["length"],
                    colors="#8B4513", linewidths=2.0, alpha=0.5, zorder=1,
                )
            else:
                x = spec["offset"]
                ax.fill_betweenx(
                    [spec["start_coord"], spec["start_coord"] + spec["length"]],
                    x - hw, x + hw,
                    color="#f5f0e1", alpha=0.25, zorder=0,
                )
                ax.vlines(
                    [x + hw, x - hw],
                    spec["start_coord"], spec["start_coord"] + spec["length"],
                    colors="#8B4513", linewidths=2.0, alpha=0.5, zorder=1,
                )

        for i, path in enumerate(self.paths):
            px, py = path.get_all_xy()
            ax.plot(
                px, py, "--", color=ROBOT_COLORS[i], lw=1.5, alpha=0.45,
                label=f"{ROBOT_LABELS[i]} ref",
            )
            p_start = path.position(0)
            p_goal = path.position(path.total_length)
            ax.plot(p_start[0], p_start[1], "s", color=ROBOT_COLORS[i], ms=8, zorder=5)
            ax.plot(p_goal[0], p_goal[1], "*", color=ROBOT_COLORS[i], ms=14, zorder=5)

        for i in range(self.n_robots):
            if len(self._rtraj[i]) > 1:
                rt = np.array(self._rtraj[i])
                ax.plot(
                    rt[:, 0], rt[:, 1], "-", color=ROBOT_COLORS[i],
                    lw=2, alpha=0.7, label=f"{ROBOT_LABELS[i]} traj",
                )

        al = 0.5
        for i in range(self.n_robots):
            ax.add_patch(plt.Circle(
                (self.rx[i], self.ry[i]), c["robot_radius"],
                color=ROBOT_COLORS[i], alpha=0.75, zorder=10,
            ))
            ax.arrow(
                self.rx[i], self.ry[i],
                al * np.cos(self.rtheta[i]), al * np.sin(self.rtheta[i]),
                head_width=0.15, head_length=0.1,
                fc=ROBOT_COLORS[i], ec=ROBOT_COLORS[i], zorder=11,
            )
            ax.add_patch(plt.Circle(
                (self.rx[i], self.ry[i]), c["safety_dist"],
                fill=False, ls="--", color=ROBOT_COLORS[i], alpha=0.25, zorder=9,
            ))

        for a, b in self._collision_pairs:
            ax.plot(
                [self.rx[a], self.rx[b]], [self.ry[a], self.ry[b]],
                "r-", lw=3, alpha=0.8, zorder=12,
            )

        elapsed_s = self.steps * c["dt"]
        n_done = sum(1 for i in range(self.n_robots) if self._robot_reached_goal(i))
        title = (
            f"t={elapsed_s:.1f}s  |  Step {self.steps}  |  "
            f"Goals: {n_done}/{self.n_robots}  |  "
            f"Collisions: {len(self._collision_pairs)}"
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


def _should_use_rl(env: MultiRobotEnv, robot_idx: int) -> bool:
    obs = env._obs_for_robot(robot_idx)
    risk = float(obs[-1])
    if risk < 0.5:
        return False

    c = env.cfg
    other_idx, _ = env._nearest_other_robot(robot_idx)
    if other_idx is None:
        return False

    cr, sr = np.cos(env.rtheta[robot_idx]), np.sin(env.rtheta[robot_idx])
    dx = env.rx[other_idx] - env.rx[robot_idx]
    dy = env.ry[other_idx] - env.ry[robot_idx]
    h_forward = cr * dx + sr * dy
    h_lateral = -sr * dx + cr * dy

    return (
        h_forward > -c.get("rl_trigger_behind_dist", 0.5)
        and h_forward < c.get("rl_trigger_forward_dist", 7.0)
        and abs(h_lateral) < c.get("rl_trigger_lateral_dist", 2.2)
    )


def run_multi_robot_demo(
    rl_model=None,
    config: dict | None = None,
    seed: int = 42,
    render: bool = True,
    save_video: str | None = None,
    policy_mode: str = "path",
) -> dict:
    """Run a four-robot crossing demonstration."""
    env = MultiRobotEnv(config=config, render_mode="human" if render else None)
    report_path: str | None = None
    if save_video:
        _run_dir, save_video, report_path = _resolve_run_output_paths(save_video)
        os.makedirs(_run_dir, exist_ok=True)
        env.start_recording()

    obs, info = env.reset(seed=seed)

    mode = str(policy_mode).lower()
    if mode == "hybrid" and rl_model is None:
        print("policy=hybrid requires --model; fallback to policy=path")
        mode = "path"
    if mode not in {"hybrid", "path", "stop_corridor"}:
        raise ValueError(f"Unknown policy_mode: {policy_mode}")

    print("Starting multi-robot run (4 robots, no humans)")
    print(_format_layout_summary(env.path_specs))

    stopped = [False] * env.n_robots
    done = False
    while not done:
        actions = np.zeros((env.n_robots, 2), dtype=np.float32)
        for i in range(env.n_robots):
            obs_i = env._obs_for_robot(i)
            if mode == "hybrid" and rl_model is not None and _should_use_rl(env, i):
                action_i, _ = rl_model.predict(obs_i, deterministic=True)
            elif mode == "stop_corridor":
                other_idx, dist = env._nearest_other_robot(i)
                in_corridor = (
                    other_idx is not None
                    and env._in_corridor(i, env.rx[other_idx], env.ry[other_idx])
                )
                if in_corridor:
                    action_i = np.array([0.0, 0.0], dtype=np.float32)
                    if not stopped[i]:
                        print(f"  Robot {i}: stop — robot {other_idx} in corridor")
                    stopped[i] = True
                else:
                    action_i = obs_to_path_goal(obs_i, env.cfg, lookahead_idx=3)
                    if stopped[i]:
                        print(f"  Robot {i}: resume")
                    stopped[i] = False
            else:
                action_i = obs_to_path_goal(obs_i, env.cfg, lookahead_idx=3)
            actions[i] = action_i

        obs, reward, terminated, truncated, info = env.step(actions.flatten())
        done = terminated or truncated
        env.render()

        if info.get("robot_collision"):
            for a, b in info.get("collision_pairs", []):
                print(f"  COLLISION: Robot {a} & Robot {b}")

    elapsed_s = _elapsed_seconds(info["step"], env.cfg["dt"])
    if info.get("all_goals_reached"):
        print(f"\nAll goals reached in {info['step']} steps ({elapsed_s:.1f} s)")
    elif info.get("robot_collision"):
        print(f"\nCollision at step {info['step']} ({elapsed_s:.1f} s)")
    elif info.get("timeout"):
        print(f"\nTimeout at {info['step']} steps ({elapsed_s:.1f} s)")
    elif info.get("out_of_bounds"):
        print(f"\nOut of bounds at {info['step']} steps ({elapsed_s:.1f} s)")

    progress = info.get("robot_progress", [])
    print(f"\n--- Final Report (total time: {elapsed_s:.1f} s) ---")
    for i, prog in enumerate(progress):
        print(f"  Robot {i} ({ROBOT_LABELS[i]}): {prog * 100:.0f}% progress")

    if render or env._recording:
        env.render()
        ax = env._ax
        if ax is not None:
            if info.get("all_goals_reached"):
                tag, color = "ALL GOALS REACHED", "#388e3c"
            elif info.get("robot_collision"):
                tag, color = "COLLISION", "#d32f2f"
            else:
                tag, color = "INCOMPLETE", "#f57c00"
            ax.text(
                0.5, 0.92, tag,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=20, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4", fc=color, alpha=0.85),
                zorder=30,
            )
            if env._recording:
                env._fig.canvas.draw()
                result_frame = np.asarray(
                    env._fig.canvas.buffer_rgba()
                )[..., :3].copy()
                for _ in range(15):
                    env._frames.append(result_frame)

    if save_video:
        env.stop_recording(save_video)

    if report_path:
        saved_report = _write_run_report(
            report_path,
            steps=info["step"],
            dt=env.cfg["dt"],
            all_goals_reached=info.get("all_goals_reached", False),
            collision_pairs=info.get("collision_pairs", []),
            min_pairwise_dist=info.get("min_pairwise_dist", float("inf")),
            robot_progress=progress,
            policy_mode=mode,
            seed=seed,
            video_path=save_video,
        )
        print(f"Report saved -> {saved_report}")

    env.close()
    return {
        "steps": info["step"],
        "elapsed_s": elapsed_s,
        "all_goals_reached": info.get("all_goals_reached", False),
        "collision_pairs": info.get("collision_pairs", []),
        "robot_progress": progress,
        "video_path": save_video,
        "report_path": report_path,
    }


if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="Multi-robot simulator (4 robots, no humans)")
    pa.add_argument("--no-render", action="store_true")
    pa.add_argument(
        "--save-video", type=str, default=None, metavar="PATH",
        help="Save video + report under Evaluation_video/multi_robot/<run_name>/",
    )
    pa.add_argument("--path-length", type=float, default=35.0)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--model", type=str, default=None, help="Path to trained RL model (.zip)")
    pa.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac"])
    pa.add_argument(
        "--policy",
        type=str,
        default="path",
        choices=["hybrid", "path", "stop_corridor"],
        help="Control policy: hybrid | path | stop_corridor",
    )
    args = pa.parse_args()

    cfg = {"path_length": args.path_length}

    rl = None
    if args.model:
        from stable_baselines3 import PPO, SAC

        AlgoCls = {"ppo": PPO, "sac": SAC}[args.algo.lower()]
        rl = AlgoCls.load(args.model)

    run_multi_robot_demo(
        rl_model=rl,
        config=cfg,
        seed=args.seed,
        render=not args.no_render,
        save_video=args.save_video,
        policy_mode=args.policy,
    )
