from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_RESULT_COLORS = {
    "COLLISION": "#d32f2f",
    "SUCCESS": "#388e3c",
    "TIMEOUT": "#f57c00",
}


def start_recording(env) -> None:
    """Begin collecting frames for video export."""
    env._recording = True
    env._frames = []


def stop_recording(env, path: str = "episode.mp4", fps: int = 10) -> str | None:
    """Save collected frames to video and stop recording."""
    env._recording = False
    if not env._frames:
        print("No frames to save.")
        return None

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    im = ax.imshow(env._frames[0])

    def _update(i):
        im.set_data(env._frames[i])
        return [im]

    ani = FuncAnimation(
        fig,
        _update,
        frames=len(env._frames),
        interval=1000 // fps,
        blit=True,
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

    n_frames = len(env._frames)
    plt.close(fig)
    env._frames = []

    if saved_path:
        print(f"Video saved → {saved_path}  ({n_frames} frames)")
    return saved_path


def render_env(env):
    if env.render_mode is None and not env._recording:
        return None

    if env._fig is None:
        if env.render_mode == "human":
            plt.ion()
        env._fig, env._ax = plt.subplots(figsize=(10, 8))

    ax = env._ax
    ax.clear()

    c = env.cfg
    ms = c["map_size"]
    ax.set_xlim(-1, ms + 1)
    ax.set_ylim(-1, ms + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    px, py = env.path.get_all_xy()
    ax.plot(px, py, "b--", lw=1.5, alpha=0.4, label="Reference path")

    p_start = env.path.position(0)
    p_goal = env.path.position(env.path.total_length)
    ax.plot(p_start[0], p_start[1], "s", color="green", ms=12, label="Start", zorder=5)
    ax.plot(p_goal[0], p_goal[1], "*", color="red", ms=16, label="Goal", zorder=5)

    s_lo = env.cur_s
    s_hi = min(env.cur_s + c["corridor_len"], env.path.total_length)
    s_arr = np.linspace(s_lo, s_hi, 40)
    left, right = [], []
    for sv in s_arr:
        p = env.path.position(sv)
        n = env.path.normal(sv)
        left.append(p + c["corridor_w"] * n)
        right.append(p - c["corridor_w"] * n)
    poly = np.array(left + right[::-1])
    ax.fill(poly[:, 0], poly[:, 1], alpha=0.12, color="orange", label="Corridor")

    if len(env._rtraj) > 1:
        rt = np.array(env._rtraj)
        ax.plot(rt[:, 0], rt[:, 1], "g-", lw=2, alpha=0.7, label="Robot traj")

    if env._human_visible and len(env._htraj) > 1:
        ht = np.array(env._htraj)
        ax.plot(ht[:, 0], ht[:, 1], "r-", lw=1.5, alpha=0.5, label="Human traj")

    ax.add_patch(plt.Circle((env.rx, env.ry), c["robot_radius"], color="green", alpha=0.7))
    al = 0.5
    ax.arrow(
        env.rx,
        env.ry,
        al * np.cos(env.rtheta),
        al * np.sin(env.rtheta),
        head_width=0.15,
        head_length=0.1,
        fc="darkgreen",
        ec="darkgreen",
    )

    ax.add_patch(
        plt.Circle((env.rx, env.ry), c["safety_dist"], fill=False, ls="--", color="gold", alpha=0.4)
    )

    if env._human_visible:
        ax.add_patch(plt.Circle((env.hx, env.hy), c["human_radius"], color="red", alpha=0.7))
        ax.arrow(
            env.hx,
            env.hy,
            env.hvx * 0.8,
            env.hvy * 0.8,
            head_width=0.1,
            head_length=0.08,
            fc="darkred",
            ec="darkred",
        )

    if env._goals:
        g = env._goals[-1]
        ax.plot(g[0], g[1], "mx", ms=12, mew=3, label="Local goal")

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Step {env.steps}  |  v={env.rv:.2f} m/s  |  {env._h_behav}")

    if env.render_mode == "human":
        env._fig.canvas.draw_idle()
        env._fig.canvas.flush_events()
        try:
            plt.pause(0.01)
        except Exception:
            pass

    if env._recording or env.render_mode == "rgb_array":
        env._fig.canvas.draw()
        frame = np.asarray(env._fig.canvas.buffer_rgba())[..., :3]
        if env._recording:
            env._frames.append(frame.copy())
        if env.render_mode == "rgb_array":
            return frame

    return None


def close_render(env):
    if env._fig is not None:
        plt.close(env._fig)
        env._fig = None
        env._ax = None


def result_tag(info: dict) -> str:
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


def show_result(env, tag: str, ret: float, steps: int, wait: bool = True):
    """Render the final frame with a result banner."""
    if env.render_mode is None and not env._recording:
        return

    render_env(env)

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
            0.5, 0.02,
            "press any key or close window to continue",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="#888888",
        )

    env._fig.canvas.draw_idle()
    env._fig.canvas.flush_events()

    if env._recording:
        env._fig.canvas.draw()
        result_frame = np.asarray(env._fig.canvas.buffer_rgba())[..., :3].copy()
        for _ in range(10):
            env._frames.append(result_frame)

    if wait and env.render_mode is not None:
        plt.ioff()
        try:
            plt.waitforbuttonpress()
        except Exception:
            pass
        plt.ion()