"""
Costmap-aware robot visualization.

A robot travels along a horizontal track while a dynamic costmap shows
crowd density in the background.  When the robot detects high density
ahead, it slows down; when the path clears, it resumes full speed.

Usage:
    python visualization.py                  # live window
    python visualization.py --save out.mp4   # save video (requires ffmpeg)
    python visualization.py --save out.gif   # save gif  (requires Pillow)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.animation import FuncAnimation


# ======================================================================
# Dynamic costmap
# ======================================================================

class DynamicCostmap:
    """2D costmap with wandering Gaussian blobs simulating crowd density."""

    def __init__(
        self,
        width: float = 20.0,
        height: float = 10.0,
        resolution: float = 0.1,
        n_blobs: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self.width = width
        self.height = height
        self.res = resolution
        self.nx = int(width / resolution)
        self.ny = int(height / resolution)

        xs = np.linspace(0, width, self.nx)
        ys = np.linspace(0, height, self.ny)
        self._gx, self._gy = np.meshgrid(xs, ys)

        rng = rng or np.random.default_rng(0)
        self.blobs: list[dict] = []
        for _ in range(n_blobs):
            self.blobs.append(self._random_blob(rng))

        self.grid = np.zeros((self.ny, self.nx), dtype=np.float64)

    def _random_blob(self, rng: np.random.Generator) -> dict:
        return dict(
            cx=float(rng.uniform(2, self.width - 2)),
            cy=float(rng.uniform(1, self.height - 1)),
            sigma=float(rng.uniform(0.8, 1.8)),
            amp=float(rng.uniform(0.5, 1.0)),
            vx=float(rng.uniform(-0.08, 0.08)),
            vy=float(rng.uniform(-0.04, 0.04)),
            phase=float(rng.uniform(0, 2 * np.pi)),
            freq=float(rng.uniform(0.02, 0.08)),
        )

    def update(self, t: float, dt: float, rng: np.random.Generator) -> None:
        self.grid[:] = 0
        for b in self.blobs:
            b["cx"] += b["vx"] * dt
            b["cy"] += b["vy"] * dt

            if b["cx"] < 0.5 or b["cx"] > self.width - 0.5:
                b["vx"] *= -1
            if b["cy"] < 0.5 or b["cy"] > self.height - 0.5:
                b["vy"] *= -1

            pulse = 0.5 + 0.5 * np.sin(b["freq"] * t * 2 * np.pi + b["phase"])
            amp = b["amp"] * pulse

            dx = self._gx - b["cx"]
            dy = self._gy - b["cy"]
            self.grid += amp * np.exp(-(dx ** 2 + dy ** 2) / (2 * b["sigma"] ** 2))

        np.clip(self.grid, 0, 1, out=self.grid)

    def query_ahead(
        self, rx: float, ry: float, look_len: float, lane_half_w: float,
    ) -> float:
        """Return the max costmap value in a rectangle ahead of (rx, ry)."""
        x_lo = max(rx, 0)
        x_hi = min(rx + look_len, self.width)
        y_lo = max(ry - lane_half_w, 0)
        y_hi = min(ry + lane_half_w, self.height)

        ix_lo = int(x_lo / self.res)
        ix_hi = int(x_hi / self.res)
        iy_lo = int(y_lo / self.res)
        iy_hi = int(y_hi / self.res)

        ix_lo = np.clip(ix_lo, 0, self.nx - 1)
        ix_hi = np.clip(ix_hi, 0, self.nx)
        iy_lo = np.clip(iy_lo, 0, self.ny - 1)
        iy_hi = np.clip(iy_hi, 0, self.ny)

        region = self.grid[iy_lo:iy_hi, ix_lo:ix_hi]
        if region.size == 0:
            return 0.0
        return float(np.max(region))


# ======================================================================
# Robot on horizontal track
# ======================================================================

class TrackRobot:
    """Simple robot that moves along y = track_y, speed modulated by costmap."""

    def __init__(
        self,
        track_y: float = 5.0,
        start_x: float = 1.0,
        goal_x: float = 19.0,
        max_speed: float = 1.5,
        min_speed: float = 0.15,
        look_ahead: float = 4.0,
        lane_half_w: float = 1.2,
        slow_thresh: float = 0.25,
    ):
        self.start_x = start_x
        self.goal_x = goal_x
        self.x = start_x
        self.y = track_y
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.speed = max_speed
        self.look_ahead = look_ahead
        self.lane_half_w = lane_half_w
        self.slow_thresh = slow_thresh
        self.trail: list[tuple[float, float, float]] = []
        self.arrived = False

    def step(self, costmap: DynamicCostmap, dt: float) -> None:
        if self.arrived:
            self.speed = 0.0
            return

        crowd = costmap.query_ahead(self.x, self.y, self.look_ahead, self.lane_half_w)

        if crowd > self.slow_thresh:
            ratio = np.clip((crowd - self.slow_thresh) / (1.0 - self.slow_thresh), 0, 1)
            target_v = self.max_speed * (1.0 - ratio) + self.min_speed * ratio
        else:
            target_v = self.max_speed

        alpha = 0.15
        self.speed += alpha * (target_v - self.speed)
        self.speed = float(np.clip(self.speed, self.min_speed, self.max_speed))

        self.x += self.speed * dt
        self.trail.append((self.x, self.y, self.speed))

        if self.x >= self.goal_x:
            self.x = self.goal_x
            self.arrived = True
            self.speed = 0.0

    @property
    def crowd_ahead(self) -> float:
        """Last-queried crowd value (for display)."""
        if not self.trail:
            return 0.0
        return 0.0


# ======================================================================
# Visualization
# ======================================================================

def run_visualization(save_path: str | None = None, total_time: float = 40.0):
    if save_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        _plt.switch_backend("Agg")

    dt = 0.1
    render_dt = 0.2 if save_path else 0.1
    sim_steps_per_frame = max(1, int(render_dt / dt))
    n_frames = int(total_time / render_dt)

    rng = np.random.default_rng(42)
    costmap = DynamicCostmap(width=20, height=10, resolution=0.1, n_blobs=5, rng=rng)
    start_x, goal_x = 1.0, 19.0
    robot = TrackRobot(
        track_y=5.0, start_x=start_x, goal_x=goal_x,
        max_speed=1.5, look_ahead=4.0, lane_half_w=1.2,
    )

    fig, (ax_main, ax_speed) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.25},
    )

    costmap.update(0, dt, rng)
    im = ax_main.imshow(
        costmap.grid, origin="lower", cmap="YlOrRd",
        extent=[0, costmap.width, 0, costmap.height],
        vmin=0, vmax=1, aspect="equal", alpha=0.75,
    )
    cbar = fig.colorbar(im, ax=ax_main, label="Crowd density", shrink=0.8, pad=0.02)

    ax_main.axhline(robot.y, color="steelblue", ls="--", lw=1.5, alpha=0.5, label="Track")

    ax_main.plot(start_x, robot.y, "s", color="limegreen", ms=14, zorder=6, label="Start")
    ax_main.plot(goal_x, robot.y, "*", color="gold", ms=18, mec="darkorange", mew=1.5, zorder=6, label="Goal")

    robot_circle = Circle((robot.x, robot.y), 0.3, fc="dodgerblue", ec="navy", lw=2, zorder=5)
    ax_main.add_patch(robot_circle)

    detect_rect = Rectangle(
        (robot.x, robot.y - robot.lane_half_w),
        robot.look_ahead, robot.lane_half_w * 2,
        fc="cyan", ec="darkcyan", lw=1.5, alpha=0.18, zorder=4,
    )
    ax_main.add_patch(detect_rect)

    trail_line, = ax_main.plot([], [], "o-", color="navy", ms=2, lw=1, alpha=0.4, label="Trail")

    ax_main.set_xlim(0, costmap.width)
    ax_main.set_ylim(0, costmap.height)
    ax_main.set_xlabel("X (m)")
    ax_main.set_ylabel("Y (m)")
    ax_main.legend(loc="upper right", fontsize=9)

    title_text = ax_main.set_title("", fontsize=13, fontweight="bold")

    speed_data: list[float] = []
    crowd_data: list[float] = []
    time_data: list[float] = []
    speed_line, = ax_speed.plot([], [], color="dodgerblue", lw=2, label="Speed")
    crowd_line, = ax_speed.plot([], [], color="orangered", lw=2, ls="--", label="Crowd ahead")
    ax_speed.set_xlim(0, total_time)
    ax_speed.set_ylim(0, 1.8)
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Value")
    ax_speed.legend(loc="upper right", fontsize=9)
    ax_speed.grid(True, alpha=0.3)

    def _update(frame: int):
        t = frame * render_dt
        for _ in range(sim_steps_per_frame):
            costmap.update(t, dt, rng)
            robot.step(costmap, dt)

        crowd = costmap.query_ahead(robot.x, robot.y, robot.look_ahead, robot.lane_half_w)

        im.set_data(costmap.grid)

        robot_circle.set_center((robot.x, robot.y))
        detect_rect.set_xy((robot.x, robot.y - robot.lane_half_w))

        if len(robot.trail) > 1:
            tx = [p[0] for p in robot.trail[-80:]]
            ty = [p[1] for p in robot.trail[-80:]]
            trail_line.set_data(tx, ty)

        view_margin = 3.0
        view_lo = max(0, robot.x - view_margin)
        view_hi = view_lo + 15.0
        if view_hi > costmap.width:
            view_hi = costmap.width
            view_lo = max(0, view_hi - 15.0)
        ax_main.set_xlim(view_lo, view_hi)
        ax_main.set_ylim(0, costmap.height)

        if robot.arrived:
            status = "ARRIVED"
            color = "#1565c0"
        elif crowd > robot.slow_thresh:
            status = "SLOWING"
            color = "#d32f2f"
        else:
            status = "CRUISING"
            color = "#388e3c"

        progress = (robot.x - start_x) / (goal_x - start_x) * 100
        title_text.set_text(
            f"t = {t:.1f}s  |  speed = {robot.speed:.2f} m/s  |  "
            f"crowd = {crowd:.2f}  |  progress = {progress:.0f}%  |  [{status}]"
        )
        title_text.set_color(color)

        time_data.append(t)
        speed_data.append(robot.speed)
        crowd_data.append(crowd)
        speed_line.set_data(time_data, speed_data)
        crowd_line.set_data(time_data, crowd_data)
        if time_data:
            t_lo = max(0, time_data[-1] - 20)
            ax_speed.set_xlim(t_lo, t_lo + 20)

        return [im, robot_circle, detect_rect, trail_line, title_text,
                speed_line, crowd_line]

    fps = int(1 / render_dt)
    ani = FuncAnimation(fig, _update, frames=n_frames, interval=int(render_dt * 1000), blit=False)

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
    pa = argparse.ArgumentParser(description="Costmap-aware robot visualization")
    pa.add_argument("--save", type=str, default=None, metavar="PATH",
                    help="Save animation (e.g. out.mp4 or out.gif)")
    pa.add_argument("--time", type=float, default=40.0,
                    help="Total simulation time in seconds (default: 40)")
    args = pa.parse_args()
    run_visualization(save_path=args.save, total_time=args.time)
