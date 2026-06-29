"""Step A demo animation (build_animation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, Rectangle

from ..tools.factory_map import GOALS, MAP_BOUNDS, OBSTACLES
from ..tools.prediction import PredictorConfig, IntentParticlePredictor
from ..tools.scenario import make_workers
from .render_common import _draw_factory, _kde_heatmap


def build_animation(
    output_path: Path | None,
    num_frames: int = 80,
    num_workers: int = 2,
    preview: bool = False,
    seed: int = 7,
):
    cfg = PredictorConfig(seed=seed)
    workers = make_workers(num_frames, cfg.dt, num_workers)
    predictors = [
        IntentParticlePredictor(
            GOALS, OBSTACLES, cfg,
            rng=np.random.default_rng(seed + 17 * i),
        )
        for i in range(len(workers))
    ]

    # ----- figure layout: map on left, intent bars on right -----
    fig = plt.figure(figsize=(15, 8.2), dpi=120)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3.4, 1.0],
        height_ratios=[14.0, 1.0],
        wspace=0.22, hspace=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    # Bottom legend only spans the map column so it cannot collide with the
    # x-axis label of the intent panel on the right.
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")

    _draw_factory(ax)

    # KDE grid (coarse for speed).
    grid_x = np.linspace(MAP_BOUNDS[0], MAP_BOUNDS[1], 80)
    grid_y = np.linspace(MAP_BOUNDS[2], MAP_BOUNDS[3], 48)
    heat_im = ax.imshow(
        np.zeros((len(grid_y), len(grid_x))),
        extent=(MAP_BOUNDS[0], MAP_BOUNDS[1], MAP_BOUNDS[2], MAP_BOUNDS[3]),
        origin="lower", cmap="Oranges", alpha=0.45, vmin=0.0, vmax=1.0,
        zorder=1,
    )

    # Per-worker visual handles.
    w_artists = []
    for w in workers:
        hist_line, = ax.plot([], [], "-", lw=2.0, color=w["color"],
                             label=f"{w['name']} observed", zorder=5)
        dot = Circle((0, 0), 0.22, facecolor=w["color"],
                     edgecolor="white", lw=1.0, zorder=6)
        ax.add_patch(dot)
        mean_line, = ax.plot([], [], "-", lw=2.2, color=w["color"],
                             alpha=0.75, zorder=4,
                             label=f"{w['name']} predicted mean")
        # One faint line per candidate goal for top-K modes.
        mode_lines = [
            ax.plot([], [], "--", lw=1.3, color=w["color"], alpha=0.0, zorder=3)[0]
            for _ in range(len(GOALS))
        ]
        # 95 % uncertainty ellipses along the horizon.
        ellipses_h = []
        for _ in range(cfg.horizon_steps):
            e = Ellipse((0, 0), 0.1, 0.1, angle=0, fill=False,
                        edgecolor=w["color"], alpha=0.0, lw=1.0, zorder=3)
            ax.add_patch(e)
            ellipses_h.append(e)
        w_artists.append(dict(
            hist=hist_line, dot=dot, mean=mean_line,
            modes=mode_lines, ellipses=ellipses_h,
        ))

    # Anchor the info box at the top-left of the map where there is empty
    # space above the obstacles, so it does not occlude any workstation label.
    info_text = ax.text(
        MAP_BOUNDS[0] + 0.4, MAP_BOUNDS[3] - 0.3, "",
        fontsize=9, color="#222",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        zorder=8,
        va="top", ha="left",
    )

    # Build a single horizontal legend at the bottom, outside the map area.
    legend_handles = []
    legend_labels = []
    # Goal marker proxy (matches the actual scatter)
    legend_handles.append(plt.Line2D(
        [0], [0], marker="*", color="w", markerfacecolor="#6a3d9a",
        markeredgecolor="white", markersize=12, linestyle="None",
    ))
    legend_labels.append("Candidate worker goals")
    # Heatmap proxy
    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor="#fdae6b", alpha=0.7))
    legend_labels.append("5 s occupancy heatmap (KDE)")
    # Mode-trajectory proxy
    legend_handles.append(plt.Line2D([0], [0], linestyle="--", color="#555555", lw=1.5))
    legend_labels.append("Per-goal predicted mode (alpha = P(goal))")
    # Ellipse proxy
    legend_handles.append(plt.Line2D([0], [0], marker="o", color="#555555",
                                     markerfacecolor="none", markersize=10,
                                     linestyle="None"))
    legend_labels.append("95% uncertainty ellipses along 5 s horizon")
    # One entry per worker (observed + predicted-mean color)
    for w in workers:
        legend_handles.append(plt.Line2D([0], [0], color=w["color"], lw=2.4))
        legend_labels.append(f"{w['name']}: observed (solid) + mean prediction")

    ax_legend.legend(
        legend_handles, legend_labels,
        loc="center", ncol=3, frameon=True,
        framealpha=0.95, edgecolor="#cccccc", fontsize=9,
    )

    # ----- intent probability side panel (uses data coordinates per worker) -----
    ax_bar.set_title("Intent probabilities", fontsize=11, pad=8)
    ax_bar.set_xlim(0.0, 1.0)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_xticks([0.0, 0.5, 1.0])
    ax_bar.set_xlabel("P(goal | history)")
    ax_bar.set_yticks([])
    ax_bar.set_facecolor("#fafafa")
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    G = len(GOALS)
    W = len(workers)
    margin_top = 0.04
    # Leave more room at the bottom so the xlabel never collides with the bars.
    margin_bottom = 0.12
    gap_between_workers = 0.04
    usable = 1.0 - margin_top - margin_bottom - gap_between_workers * max(W - 1, 0)
    block_height = usable / W                          # height per worker block
    bar_height = block_height / (G + 1) * 0.85         # leave room for header
    header_height = block_height - G * bar_height
    label_left = 0.05
    bar_left = 0.18
    bar_full_width = 0.72
    bar_text_x = bar_left + bar_full_width + 0.02

    bar_groups = []
    for wi, w in enumerate(workers):
        block_top = 1.0 - margin_top - wi * (block_height + gap_between_workers)
        # Worker header label
        ax_bar.text(
            label_left, block_top - header_height * 0.55, w["name"],
            fontsize=10.5, color=w["color"], fontweight="bold",
            transform=ax_bar.transAxes, va="center",
        )
        bars = []
        for gi in range(G):
            row_top = block_top - header_height - gi * bar_height
            y = row_top - bar_height * 0.9            # bottom of bar
            rect = Rectangle(
                (bar_left, y), 0.0, bar_height * 0.78,
                facecolor=w["color"], alpha=0.7,
                transform=ax_bar.transAxes,
            )
            ax_bar.add_patch(rect)
            ax_bar.text(
                label_left, y + bar_height * 0.39, f"G{gi+1}",
                fontsize=8.5, color="#444",
                transform=ax_bar.transAxes, va="center",
            )
            txt = ax_bar.text(
                bar_text_x, y + bar_height * 0.39, "0.00",
                fontsize=8.5, color="#222",
                transform=ax_bar.transAxes, va="center", ha="left",
            )
            bars.append((rect, txt))
        bar_groups.append(bars)

    # ----- per-frame update -----
    def update(frame: int):
        heat = np.zeros((len(grid_y), len(grid_x)))

        for wi, (w, predictor, art) in enumerate(zip(workers, predictors, w_artists)):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]

            out = predictor.rollout(obs)
            belief = out["belief"]
            particles = out["particles"]
            mean_traj = out["mean"]
            mode_traj = out["mode_traj"]
            ellipses = out["ellipses"]

            art["hist"].set_data(w["truth"][: frame + 1, 0],
                                 w["truth"][: frame + 1, 1])
            art["dot"].center = tuple(w["truth"][frame])
            art["mean"].set_data(mean_traj[:, 0], mean_traj[:, 1])

            for gi, line in enumerate(art["modes"]):
                line.set_data(mode_traj[gi, :, 0], mode_traj[gi, :, 1])
                line.set_alpha(float(np.clip(belief[gi] * 1.3, 0.05, 0.95)))
                line.set_linewidth(1.2 + 2.4 * float(belief[gi]))

            for t, e in enumerate(art["ellipses"]):
                x, y, ww, hh, ang = ellipses[t]
                e.center = (x, y)
                e.width = ww
                e.height = hh
                e.angle = ang
                e.set_alpha(0.04 + 0.18 * t / cfg.horizon_steps)

            heat += _kde_heatmap(particles, grid_x, grid_y)

            bars = bar_groups[wi]
            for gi, (rect, txt) in enumerate(bars):
                rect.set_width(bar_full_width * float(belief[gi]))
                txt.set_text(f"{belief[gi]:.2f}")

        if heat.max() > 1e-6:
            heat = heat / heat.max()
        heat_im.set_data(heat)

        info_text.set_text(
            f"frame {frame+1}/{num_frames}    "
            f"horizon {cfg.horizon_steps*cfg.dt:.1f} s    "
            f"particles/worker {cfg.num_particles}\n"
            "method: recursive Bayes intent  +  social-force particle rollout  "
            "+  KDE occupancy"
        )

        if not preview and (frame % 5 == 0 or frame == num_frames - 1):
            print(f"  rendering frame {frame+1}/{num_frames}")

        artists = [heat_im, info_text]
        for art in w_artists:
            artists += [art["hist"], art["dot"], art["mean"]]
            artists += art["modes"] + art["ellipses"]
        for bars in bar_groups:
            for rect, txt in bars:
                artists += [rect, txt]
        return artists

    anim = FuncAnimation(fig, update, frames=num_frames, interval=90, blit=False)

    if preview:
        print("Preview mode: showing live window. Close it to exit.")
        plt.show()
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(fps=12, bitrate=2400)
        anim.save(output_path, writer=writer)
        saved = output_path
    except Exception as exc:                                # pragma: no cover
        print(f"  ffmpeg unavailable ({exc}); falling back to GIF")
        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=10))
        saved = gif_path
    plt.close(fig)
    return saved
