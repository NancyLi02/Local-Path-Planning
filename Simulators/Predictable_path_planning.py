"""
Predictive Local Path Planning with Dynamic Costmap.

A robot travels along a horizontal reference track.  Crowd blobs drift
across the costmap; each blob's velocity is estimated and shown as a
prediction arrow.  When a blob threatens the robot's forward corridor,
the planner generates a smooth curved detour in the *opposite* direction
of the blob's predicted movement, then merges back onto the reference
path once clear.

Usage:
    python Predictable_path_planning.py                  # live window
    python Predictable_path_planning.py --save out.mp4   # save video
    python Predictable_path_planning.py --save out.gif   # save gif
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.animation import FuncAnimation


# ======================================================================
# Dynamic costmap with observable blob velocities
# ======================================================================

class CrowdBlob:
    """A single Gaussian crowd blob with position, size and velocity."""

    def __init__(self, cx, cy, sigma, amp, vx, vy, phase, freq):
        self.cx = cx
        self.cy = cy
        self.sigma = sigma
        self.amp = amp
        self.vx = vx
        self.vy = vy
        self.phase = phase
        self.freq = freq

    def predicted_pos(self, dt: float) -> tuple[float, float]:
        return self.cx + self.vx * dt, self.cy + self.vy * dt

    def current_amplitude(self, t: float) -> float:
        pulse = 0.5 + 0.5 * np.sin(self.freq * t * 2 * np.pi + self.phase)
        return self.amp * pulse


class PredictiveCostmap:
    """2D costmap whose blobs expose velocity for prediction."""

    def __init__(
        self,
        width: float = 24.0,
        height: float = 10.0,
        resolution: float = 0.1,
        n_blobs: int = 5,
        track_y: float = 5.0,
        rng: np.random.Generator | None = None,
    ):
        self.width = width
        self.height = height
        self.track_y = track_y
        self.res = resolution
        self.nx = int(width / resolution)
        self.ny = int(height / resolution)

        xs = np.linspace(0, width, self.nx)
        ys = np.linspace(0, height, self.ny)
        self._gx, self._gy = np.meshgrid(xs, ys)

        rng = rng or np.random.default_rng(0)
        self.blobs: list[CrowdBlob] = []
        n_on_track = min(2, n_blobs)
        for i in range(n_blobs):
            if i < n_on_track:
                self.blobs.append(self._make_blob_on_track(rng))
            else:
                self.blobs.append(self._make_blob_off_track(rng))

        self.grid = np.zeros((self.ny, self.nx), dtype=np.float64)

    def _make_blob_on_track(self, rng: np.random.Generator) -> CrowdBlob:
        """Blob that starts on/near the reference path and drifts across it."""
        return CrowdBlob(
            cx=float(rng.uniform(4, self.width - 2)),
            cy=self.track_y + float(rng.uniform(-0.5, 0.5)),
            sigma=float(rng.uniform(0.8, 1.5)),
            amp=float(rng.uniform(0.6, 1.0)),
            vx=float(rng.uniform(-0.15, 0.15)),
            vy=float(rng.choice([-1, 1])) * float(rng.uniform(0.08, 0.15)),
            phase=float(rng.uniform(0, 2 * np.pi)),
            freq=float(rng.uniform(0.02, 0.06)),
        )

    def _make_blob_off_track(self, rng: np.random.Generator) -> CrowdBlob:
        """Blob that stays away from the reference path (background noise)."""
        side = float(rng.choice([-1, 1]))
        return CrowdBlob(
            cx=float(rng.uniform(3, self.width - 3)),
            cy=self.track_y + side * float(rng.uniform(2.5, 4.0)),
            sigma=float(rng.uniform(0.8, 1.8)),
            amp=float(rng.uniform(0.3, 0.7)),
            vx=float(rng.uniform(-0.15, 0.15)),
            vy=float(rng.uniform(-0.10, 0.10)),
            phase=float(rng.uniform(0, 2 * np.pi)),
            freq=float(rng.uniform(0.02, 0.08)),
        )

    def update(self, t: float, dt: float) -> None:
        self.grid[:] = 0
        for b in self.blobs:
            b.cx += b.vx * dt
            b.cy += b.vy * dt
            if b.cx < 1.0 or b.cx > self.width - 1.0:
                b.vx *= -1
            if b.cy < 1.0 or b.cy > self.height - 1.0:
                b.vy *= -1

            amp = b.current_amplitude(t)
            dx = self._gx - b.cx
            dy = self._gy - b.cy
            self.grid += amp * np.exp(-(dx ** 2 + dy ** 2) / (2 * b.sigma ** 2))
        np.clip(self.grid, 0, 1, out=self.grid)

    def threatening_blobs(
        self, rx: float, track_y: float, look_len: float,
        path_half_w: float, t: float, amp_thresh: float = 0.2,
    ) -> list[CrowdBlob]:
        """Return blobs that overlap the reference path ahead of the robot.

        A blob is threatening only when:
          1. It is active (amplitude > threshold).
          2. It is ahead of the robot within *look_len*.
          3. Its effective radius (sigma) overlaps the reference track
             corridor of half-width *path_half_w* centred on *track_y*.
        """
        result = []
        for b in self.blobs:
            if b.current_amplitude(t) < amp_thresh:
                continue
            if b.cx < rx or b.cx > rx + look_len:
                continue
            dist_to_track = abs(b.cy - track_y)
            if dist_to_track > path_half_w + b.sigma:
                continue

            on_track = dist_to_track <= path_half_w
            if not on_track:
                offset = b.cy - track_y
                moving_away = (offset > 0 and b.vy > 0) or (offset < 0 and b.vy < 0)
                if moving_away:
                    continue

            result.append(b)
        return result


# ======================================================================
# Predictive local path planner
# ======================================================================

class PredictivePlanner:
    """Plan a smooth curved detour opposite to the crowd blob's velocity."""

    def __init__(
        self,
        track_y: float = 5.0,
        dodge_amplitude: float = 2.5,
        entry_len: float = 3.0,
        exit_len: float = 3.0,
        predict_horizon: float = 3.0,
        n_path_pts: int = 60,
    ):
        self.track_y = track_y
        self.dodge_amp = dodge_amplitude
        self.entry_len = entry_len
        self.exit_len = exit_len
        self.predict_horizon = predict_horizon
        self.n_pts = n_path_pts

    def plan(
        self, rx: float, ry: float, threat: CrowdBlob, t: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (xs, ys) arrays for the local detour path, or None."""
        pred_bx, pred_by = threat.predicted_pos(self.predict_horizon)

        if abs(threat.vy) > 0.005:
            dodge_dir = -np.sign(threat.vy)
        else:
            dodge_dir = -1.0 if threat.cy >= self.track_y else 1.0

        intensity = np.clip(threat.current_amplitude(t) / threat.amp, 0.3, 1.0)
        lateral = dodge_dir * self.dodge_amp * intensity

        y_limit_margin = 1.0
        if self.track_y + lateral > 10.0 - y_limit_margin:
            lateral = 10.0 - y_limit_margin - self.track_y
        if self.track_y + lateral < y_limit_margin:
            lateral = y_limit_margin - self.track_y

        x_blob = threat.cx
        x_start = max(rx, x_blob - self.entry_len)
        x_apex = x_blob
        x_end = x_blob + self.exit_len

        wx = np.array([x_start, (x_start + x_apex) / 2, x_apex, (x_apex + x_end) / 2, x_end])
        wy = np.array([
            ry,
            self.track_y + lateral * 0.6,
            self.track_y + lateral,
            self.track_y + lateral * 0.6,
            self.track_y,
        ])

        ts = np.linspace(0, 1, len(wx))
        cs_x = CubicSpline(ts, wx, bc_type="clamped")
        cs_y = CubicSpline(ts, wy, bc_type="clamped")

        t_fine = np.linspace(0, 1, self.n_pts)
        return cs_x(t_fine), cs_y(t_fine)


# ======================================================================
# Robot with local path tracking
# ======================================================================

class PlanningRobot:
    """Robot that follows the reference track or a local detour path."""

    def __init__(
        self,
        track_y: float = 5.0,
        start_x: float = 1.0,
        goal_x: float = 23.0,
        max_speed: float = 1.2,
        min_speed: float = 0.3,
    ):
        self.track_y = track_y
        self.start_x = start_x
        self.goal_x = goal_x
        self.x = start_x
        self.y = track_y
        self.speed = max_speed
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.arrived = False

        self.local_path: tuple[np.ndarray, np.ndarray] | None = None
        self._lp_idx = 0

        self.trail: list[tuple[float, float]] = []

    def set_local_path(self, path: tuple[np.ndarray, np.ndarray] | None) -> None:
        if path is None:
            self.local_path = None
            self._lp_idx = 0
            return
        xs, ys = path
        mask = xs >= self.x - 0.1
        if mask.sum() < 2:
            self.local_path = None
            return
        self.local_path = (xs[mask], ys[mask])
        self._lp_idx = 0

    def step(self, dt: float, crowd_level: float) -> None:
        if self.arrived:
            self.speed = 0.0
            return

        if crowd_level > 0.15:
            ratio = np.clip(crowd_level, 0, 1)
            target_v = self.max_speed * (1.0 - 0.5 * ratio)
        else:
            target_v = self.max_speed

        self.speed += 0.2 * (target_v - self.speed)
        self.speed = float(np.clip(self.speed, self.min_speed, self.max_speed))

        dx = self.speed * dt

        if self.local_path is not None:
            xs, ys = self.local_path
            while self._lp_idx < len(xs) - 1 and xs[self._lp_idx] < self.x:
                self._lp_idx += 1

            if self._lp_idx >= len(xs) - 1:
                self.local_path = None
                self._lp_idx = 0
                target_y = self.track_y
            else:
                target_y = float(np.interp(self.x + dx, xs, ys))
        else:
            target_y = self.track_y

        max_lateral_step = 1.5 * dt
        dy = np.clip(target_y - self.y, -max_lateral_step, max_lateral_step)

        self.x += dx
        self.y += dy
        self.trail.append((self.x, self.y))

        if self.x >= self.goal_x:
            self.x = self.goal_x
            self.arrived = True
            self.speed = 0.0


# ======================================================================
# Visualization
# ======================================================================

def run(save_path: str | None = None, total_time: float = 50.0):
    if save_path:
        import matplotlib
        matplotlib.use("Agg")

    dt = 0.1
    render_dt = 0.2 if save_path else 0.1
    steps_per_frame = max(1, int(render_dt / dt))
    n_frames = int(total_time / render_dt)

    rng = np.random.default_rng(12)
    track_y = 5.0
    costmap = PredictiveCostmap(width=24, height=10, resolution=0.1, n_blobs=5,
                                track_y=track_y, rng=rng)
    start_x, goal_x = 1.0, 23.0
    robot = PlanningRobot(track_y=track_y, start_x=start_x, goal_x=goal_x,
                          max_speed=1.2, min_speed=0.3)
    planner = PredictivePlanner(track_y=track_y, dodge_amplitude=2.5,
                                entry_len=3.0, exit_len=3.0, predict_horizon=3.0)
    look_ahead = 6.0
    path_half_w = 0.8

    fig, (ax_main, ax_info) = plt.subplots(
        2, 1, figsize=(15, 9),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.28},
    )

    costmap.update(0, dt)
    im = ax_main.imshow(
        costmap.grid, origin="lower", cmap="YlOrRd",
        extent=[0, costmap.width, 0, costmap.height],
        vmin=0, vmax=1, aspect="equal", alpha=0.7,
    )
    fig.colorbar(im, ax=ax_main, label="Crowd density", shrink=0.75, pad=0.02)

    ax_main.axhline(track_y, color="steelblue", ls="--", lw=1.5, alpha=0.4, label="Reference track")
    ax_main.plot(start_x, track_y, "s", color="limegreen", ms=14, zorder=6, label="Start")
    ax_main.plot(goal_x, track_y, "*", color="gold", ms=18, mec="darkorange", mew=1.5,
                 zorder=6, label="Goal")

    robot_circle = Circle((robot.x, robot.y), 0.28, fc="dodgerblue", ec="navy", lw=2, zorder=8)
    ax_main.add_patch(robot_circle)

    detect_rect = Rectangle(
        (robot.x, track_y - path_half_w), look_ahead, path_half_w * 2,
        fc="cyan", ec="darkcyan", lw=1.2, alpha=0.12, zorder=3,
    )
    ax_main.add_patch(detect_rect)

    trail_line, = ax_main.plot([], [], "-", color="navy", lw=1.8, alpha=0.5, label="Robot trail")
    plan_line, = ax_main.plot([], [], "-", color="magenta", lw=2.5, alpha=0.7, label="Local plan")

    _dynamic_artists: list = []

    ax_main.set_xlim(0, costmap.width)
    ax_main.set_ylim(0, costmap.height)
    ax_main.set_xlabel("X (m)")
    ax_main.set_ylabel("Y (m)")
    ax_main.legend(loc="upper right", fontsize=8, ncol=2)
    title_text = ax_main.set_title("", fontsize=12, fontweight="bold")

    time_data: list[float] = []
    speed_data: list[float] = []
    lat_data: list[float] = []
    speed_line, = ax_info.plot([], [], color="dodgerblue", lw=2, label="Speed (m/s)")
    lat_line, = ax_info.plot([], [], color="orchid", lw=2, ls="-", label="Lateral offset (m)")
    ax_info.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax_info.set_ylim(-3.5, 2.5)
    ax_info.set_xlabel("Time (s)")
    ax_info.legend(loc="upper right", fontsize=9)
    ax_info.grid(True, alpha=0.3)

    sim_t = [0.0]

    def _update(frame: int):
        for art in _dynamic_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        _dynamic_artists.clear()

        for _ in range(steps_per_frame):
            sim_t[0] += dt
            costmap.update(sim_t[0], dt)

            threats = costmap.threatening_blobs(
                robot.x, track_y, look_ahead, path_half_w, sim_t[0],
            )

            if threats and not robot.arrived:
                nearest = min(threats, key=lambda b: abs(b.cx - robot.x))
                path = planner.plan(robot.x, robot.y, nearest, sim_t[0])
                robot.set_local_path(path)
                crowd_level = nearest.current_amplitude(sim_t[0])
            else:
                if robot.local_path is None:
                    pass
                crowd_level = 0.0

            robot.step(dt, crowd_level)

        t = sim_t[0]

        im.set_data(costmap.grid)
        robot_circle.set_center((robot.x, robot.y))
        detect_rect.set_xy((robot.x, track_y - path_half_w))

        if len(robot.trail) > 1:
            tx = [p[0] for p in robot.trail[-120:]]
            ty = [p[1] for p in robot.trail[-120:]]
            trail_line.set_data(tx, ty)

        if robot.local_path is not None:
            px, py = robot.local_path
            plan_line.set_data(px, py)
        else:
            plan_line.set_data([], [])

        for b in costmap.blobs:
            amp_now = b.current_amplitude(t)
            if amp_now < 0.15:
                continue

            arrow_scale = 15.0
            arr = ax_main.annotate(
                "", xy=(b.cx + b.vx * arrow_scale, b.cy + b.vy * arrow_scale),
                xytext=(b.cx, b.cy),
                arrowprops=dict(arrowstyle="-|>", color="darkred", lw=2.0, mutation_scale=14),
                zorder=7,
            )
            _dynamic_artists.append(arr)

            for dt_pred in [3.0, 6.0]:
                px, py = b.predicted_pos(dt_pred)
                alpha_g = 0.25 * (1.0 - dt_pred / 8.0)
                ghost = Circle((px, py), b.sigma * 0.6, fc="red", alpha=alpha_g,
                               ec="darkred", ls=":", lw=0.8, zorder=2)
                ax_main.add_patch(ghost)
                _dynamic_artists.append(ghost)

        view_margin = 3.0
        view_lo = max(0, robot.x - view_margin)
        view_hi = view_lo + 15.0
        if view_hi > costmap.width:
            view_hi = costmap.width
            view_lo = max(0, view_hi - 15.0)
        ax_main.set_xlim(view_lo, view_hi)
        ax_main.set_ylim(0, costmap.height)

        threats_now = costmap.threatening_blobs(robot.x, track_y, look_ahead, path_half_w, t)
        if robot.arrived:
            status, color = "ARRIVED", "#1565c0"
        elif threats_now:
            status, color = "DODGING", "#d32f2f"
        else:
            status, color = "CRUISING", "#388e3c"

        progress = (robot.x - start_x) / (goal_x - start_x) * 100
        title_text.set_text(
            f"t={t:.1f}s | speed={robot.speed:.2f} m/s | "
            f"lateral={robot.y - track_y:+.2f}m | "
            f"progress={progress:.0f}% | [{status}]"
        )
        title_text.set_color(color)

        time_data.append(t)
        speed_data.append(robot.speed)
        lat_data.append(robot.y - track_y)
        speed_line.set_data(time_data, speed_data)
        lat_line.set_data(time_data, lat_data)
        if time_data:
            t_lo = max(0, time_data[-1] - 25)
            ax_info.set_xlim(t_lo, t_lo + 25)

        return []

    fps = int(1 / render_dt)
    ani = FuncAnimation(fig, _update, frames=n_frames,
                        interval=int(render_dt * 1000), blit=False)

    if save_path:
        print(f"Rendering {n_frames} frames at {fps} fps ...")
        if save_path.endswith(".gif"):
            from matplotlib.animation import PillowWriter
            ani.save(save_path, writer=PillowWriter(fps=fps), dpi=80)
        else:
            try:
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=fps, extra_args=["-preset", "ultrafast"])
                ani.save(save_path, writer=writer, dpi=80)
            except Exception:
                gif_path = save_path.rsplit(".", 1)[0] + ".gif"
                print(f"ffmpeg failed, falling back to gif → {gif_path}")
                from matplotlib.animation import PillowWriter
                ani.save(gif_path, writer=PillowWriter(fps=fps), dpi=80)
                save_path = gif_path
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser(description="Predictive local path planning visualization")
    pa.add_argument("--save", type=str, default=None, metavar="PATH",
                    help="Save animation (e.g. out.mp4 or out.gif)")
    pa.add_argument("--time", type=float, default=50.0,
                    help="Total simulation time in seconds (default: 50)")
    args = pa.parse_args()
    run(save_path=args.save, total_time=args.time)
