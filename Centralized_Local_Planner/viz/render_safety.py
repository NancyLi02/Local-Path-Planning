"""Steps A+B demo animation (build_safety_animation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Patch, Polygon, Rectangle

from ..tools.factory_map import GOALS, MAP_BOUNDS, OBSTACLES
from ..tools.prediction import PredictorConfig, IntentParticlePredictor
from ..tools.safety_inflation import (
    SafetyInflationConfig, SafetyInflationModel, DEFAULT_FRAMES_SAFETY,
)
from ..tools.geometry import safety_tube_polygon
from ..tools.scenario import make_workers
from .render_common import _rgba, _draw_factory, _kde_heatmap


# Component color palette (used for the breakdown stacked bar + legend).
COMPONENT_COLORS: dict[str, str] = dict(
    body="#4c72b0",
    latency="#dd8452",
    braking="#c44e52",
    reaction="#8172b3",
    forecast="#937860",
)

def build_safety_animation(
    output_path: Path | None,
    num_frames: int = DEFAULT_FRAMES_SAFETY,
    num_workers: int = 2,
    preview: bool = False,
    seed: int = 7,
):
    cfg = PredictorConfig(seed=seed)
    safety_cfg = SafetyInflationConfig()
    safety = SafetyInflationModel(safety_cfg)

    workers = make_workers(num_frames, cfg.dt, num_workers)
    predictors = [
        IntentParticlePredictor(
            GOALS, OBSTACLES, cfg,
            rng=np.random.default_rng(seed + 17 * i),
        )
        for i in range(len(workers))
    ]

    horizon_times = np.arange(1, cfg.horizon_steps + 1) * cfg.dt

    # ----- figure layout -----
    fig = plt.figure(figsize=(15.8, 10.2), dpi=120)
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[3.0, 1.05],
        height_ratios=[11.5, 4.0, 0.9],
        wspace=0.20, hspace=0.34,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_intent = fig.add_subplot(gs[0, 1])
    ax_rad = fig.add_subplot(gs[1, 0])
    ax_break = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")

    _draw_factory(ax)
    ax.set_title(
        "Steps A + B: worker trajectory prediction + safety inflation",
        fontsize=13, pad=10,
    )

    # ----- KDE heatmap (Step A) -----
    grid_x = np.linspace(MAP_BOUNDS[0], MAP_BOUNDS[1], 80)
    grid_y = np.linspace(MAP_BOUNDS[2], MAP_BOUNDS[3], 48)
    heat_im = ax.imshow(
        np.zeros((len(grid_y), len(grid_x))),
        extent=(MAP_BOUNDS[0], MAP_BOUNDS[1], MAP_BOUNDS[2], MAP_BOUNDS[3]),
        origin="lower", cmap="Oranges", alpha=0.40, vmin=0.0, vmax=1.0,
        zorder=1,
    )

    # ----- per-worker visual handles -----
    # Re-designed: instead of stacking many ellipses, we draw exactly THREE
    # shapes per worker for the safety inflation:
    #   * one filled "hard" swept tube (convex hull of all inflated ellipses)
    #   * one dotted "soft" outer tube (hard tube grown by ``soft_margin``)
    #   * one red NO-GO ellipse at t=0  (current forbidden region)
    # This collapses the previous 30+ ellipse stack into 3 clean shapes.
    w_artists = []
    placeholder = np.array([[0.0, 0.0], [0.0, 1e-3], [1e-3, 0.0]])
    for w in workers:
        hist_line, = ax.plot([], [], "-", lw=2.0, color=w["color"], zorder=5)
        dot = Circle((0, 0), 0.22, facecolor=w["color"],
                     edgecolor="white", lw=1.0, zorder=6)
        ax.add_patch(dot)
        # Always-on name tag pinned to the worker's current position.
        tag = ax.text(0, 0, w["name"], fontsize=8.5, color="white",
                      fontweight="bold", ha="center", va="bottom",
                      bbox=dict(facecolor=w["color"], edgecolor="white",
                                lw=0.6, pad=2, boxstyle="round,pad=0.18"),
                      zorder=7)
        mean_line, = ax.plot([], [], "-", lw=2.2, color=w["color"],
                             alpha=0.85, zorder=5)
        mode_lines = [
            ax.plot([], [], "--", lw=1.1, color=w["color"], alpha=0.0, zorder=3)[0]
            for _ in range(len(GOALS))
        ]
        # Soft slowdown tube (outer, dotted, no fill).
        soft_tube = Polygon(placeholder, closed=True,
                            facecolor="none",
                            edgecolor=_rgba(w["color"], 0.55),
                            linestyle=":", lw=1.2, zorder=2)
        ax.add_patch(soft_tube)
        # Hard safety tube (inner, filled, solid edge).
        hard_tube = Polygon(placeholder, closed=True,
                            facecolor=_rgba(w["color"], 0.18),
                            edgecolor=_rgba(w["color"], 0.75),
                            linestyle="-", lw=1.6, zorder=2)
        ax.add_patch(hard_tube)
        # Hard NO-GO contour at t=0 -- now a forward-biased TEARDROP polygon
        # rather than a symmetric ellipse, so the shape of the immediate
        # forbidden zone matches the underlying asymmetric model.
        hard_now = Polygon(placeholder, closed=True,
                           facecolor="none", edgecolor="#cc0033",
                           linestyle="-", lw=2.4, alpha=0.0, zorder=4)
        ax.add_patch(hard_now)
        # Tiny arrow showing the predicted walking direction so the user
        # immediately reads which way the teardrop is pointing.
        heading_arrow = ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=w["color"],
                            lw=1.6, alpha=0.0),
            zorder=6,
        )
        w_artists.append(dict(
            hist=hist_line, dot=dot, tag=tag, mean=mean_line, modes=mode_lines,
            soft_tube=soft_tube, hard_tube=hard_tube, hard=hard_now,
            heading=heading_arrow,
        ))

    info_text = ax.text(
        MAP_BOUNDS[0] + 0.4, MAP_BOUNDS[3] - 0.3, "",
        fontsize=9, color="#222",
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        zorder=8, va="top", ha="left",
    )

    # ----- intent probability side panel -----
    ax_intent.set_title("Intent probabilities", fontsize=11, pad=8)
    ax_intent.set_xlim(0.0, 1.0)
    ax_intent.set_ylim(0.0, 1.0)
    ax_intent.set_xticks([0.0, 0.5, 1.0])
    ax_intent.set_xlabel("P(goal | history)")
    ax_intent.set_yticks([])
    ax_intent.set_facecolor("#fafafa")
    for spine in ("top", "right"):
        ax_intent.spines[spine].set_visible(False)

    G = len(GOALS)
    W = len(workers)
    margin_top = 0.04
    margin_bottom = 0.14
    gap_between_workers = 0.04
    usable = 1.0 - margin_top - margin_bottom - gap_between_workers * max(W - 1, 0)
    block_height = usable / W
    bar_height = block_height / (G + 1) * 0.85
    header_height = block_height - G * bar_height
    label_left = 0.05
    bar_left = 0.20
    bar_full_width = 0.66
    bar_text_x = bar_left + bar_full_width + 0.02

    bar_groups = []
    entropy_texts = []
    for wi, w in enumerate(workers):
        block_top = 1.0 - margin_top - wi * (block_height + gap_between_workers)
        ax_intent.text(
            label_left, block_top - header_height * 0.50, w["name"],
            fontsize=10.5, color=w["color"], fontweight="bold",
            transform=ax_intent.transAxes, va="center",
        )
        ent_txt = ax_intent.text(
            0.95, block_top - header_height * 0.50, "",
            fontsize=8.5, color="#555",
            transform=ax_intent.transAxes, va="center", ha="right",
        )
        entropy_texts.append(ent_txt)
        bars = []
        for gi in range(G):
            row_top = block_top - header_height - gi * bar_height
            y = row_top - bar_height * 0.9
            rect = Rectangle(
                (bar_left, y), 0.0, bar_height * 0.78,
                facecolor=w["color"], alpha=0.7,
                transform=ax_intent.transAxes,
            )
            ax_intent.add_patch(rect)
            ax_intent.text(
                label_left, y + bar_height * 0.39, f"G{gi+1}",
                fontsize=8.5, color="#444",
                transform=ax_intent.transAxes, va="center",
            )
            txt = ax_intent.text(
                bar_text_x, y + bar_height * 0.39, "0.00",
                fontsize=8.5, color="#222",
                transform=ax_intent.transAxes, va="center", ha="left",
            )
            bars.append((rect, txt))
        bar_groups.append(bars)

    # ----- safety-radius vs horizon panel -----
    ax_rad.set_facecolor("#fafafa")
    ax_rad.set_xlim(0.0, horizon_times[-1])
    ax_rad.set_xlabel("Lookahead horizon  t  [s]")
    ax_rad.set_ylabel("Effective safety radius [m]")
    ax_rad.set_title("Step-B safety budget along the 5 s horizon", fontsize=11, pad=6)
    ax_rad.grid(True, alpha=0.3)

    baseline_belief = np.full(G, 1.0 / G)
    baseline_curve = np.array([
        safety.buffer_radius(t, baseline_belief, v_human=safety_cfg.v_full_walk)
        for t in horizon_times
    ])
    ax_rad.plot(horizon_times, baseline_curve, "--", color="#444",
                lw=1.4,
                label="R_buffer(t) @ uniform intent, walking speed (reference)")

    rad_lines = []
    for w in workers:
        line, = ax_rad.plot([], [], "-", lw=2.2, color=w["color"],
                            label=f"{w['name']} effective R(t)")
        rad_lines.append(line)
    # Threshold bands: green safe (R_buffer-only) | amber slowdown | red reroute.
    r_base_uniform = baseline_curve[0]
    ax_rad.axhspan(0.0, r_base_uniform, facecolor="#d9ead3", alpha=0.45)
    ax_rad.axhspan(r_base_uniform, r_base_uniform + 1.6,
                   facecolor="#fff2cc", alpha=0.45)
    ax_rad.axhspan(r_base_uniform + 1.6, 12.0,
                   facecolor="#f4cccc", alpha=0.35)
    ax_rad.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.92)
    ax_rad.set_ylim(0.0, 7.5)

    # ----- inflation breakdown panel -----
    ax_break.set_facecolor("#fafafa")
    ax_break.set_title("Inflation breakdown @ t = horizon", fontsize=11, pad=6)
    ax_break.set_xlim(0.0, 4.5)
    ax_break.set_ylim(-0.5, 0.5)
    ax_break.set_yticks([])
    ax_break.set_xlabel("Radius contribution [m]")
    for spine in ("top", "right"):
        ax_break.spines[spine].set_visible(False)

    component_order = ["body", "latency", "braking", "reaction", "forecast"]
    break_rects, break_labels = [], []
    for name in component_order:
        rect = Rectangle((0.0, -0.20), 0.0, 0.40,
                         facecolor=COMPONENT_COLORS[name],
                         alpha=0.92, edgecolor="white", lw=1.0)
        ax_break.add_patch(rect)
        lbl = ax_break.text(0.0, 0.0, "", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold",
                            clip_on=False)
        break_rects.append(rect)
        break_labels.append(lbl)

    ax_break.legend(
        handles=[Patch(facecolor=COMPONENT_COLORS[n], label=n) for n in component_order],
        loc="lower right", fontsize=7.5, ncol=3, framealpha=0.92,
    )
    break_subtitle = ax_break.text(
        0.02, 0.92, "", transform=ax_break.transAxes,
        ha="left", va="top", fontsize=8.5, color="#444",
    )

    # ----- shared bottom legend -----
    legend_handles = [
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#6a3d9a", markeredgecolor="white",
                   markersize=12, linestyle="None"),
        Rectangle((0, 0), 1, 1, facecolor="#fdae6b", alpha=0.7),
        plt.Line2D([0], [0], linestyle="--", color="#888", lw=1.4),
        Patch(facecolor=_rgba("#888888", 0.20),
              edgecolor="#888", linewidth=1.6),
        plt.Line2D([0], [0], linestyle=":", color="#888", lw=1.6),
        plt.Line2D([0], [0], color="#cc0033", lw=2.4),
        plt.Line2D([0], [0], color="#444", lw=1.6,
                   marker=">", markersize=8, markevery=[1]),
    ]
    legend_labels = [
        "Candidate goals",
        "Step A: 5 s occupancy KDE",
        "Step A: per-goal mode trajectory",
        "Step B: 5 s safety tube (swept hard zone, teardrop hull)",
        "Step B: soft slowdown tube",
        "Step B: forward-biased NO-GO teardrop @ t=0",
        "Worker walking direction (drives the teardrop)",
    ]
    for w in workers:
        legend_handles.append(plt.Line2D([0], [0], color=w["color"], lw=2.4))
        legend_labels.append(w["name"])
    ax_leg.legend(legend_handles, legend_labels, loc="center", ncol=4,
                  frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=9)

    # ----- per-frame update -----
    def update(frame: int):
        heat = np.zeros((len(grid_y), len(grid_x)))

        worker_outs = []
        for w, predictor in zip(workers, predictors):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            worker_outs.append(predictor.rollout(obs))

        ent_norms = [
            SafetyInflationModel.normalized_entropy(o["belief"])
            for o in worker_outs
        ]
        focus_wi = int(np.argmax(ent_norms))

        for wi, (w, out, art) in enumerate(zip(workers, worker_outs, w_artists)):
            belief = out["belief"]
            particles = out["particles"]
            mean_traj = out["mean"]
            mode_traj = out["mode_traj"]
            ellipses = out["ellipses"]

            # Per-step worker speed AND heading are taken from the Step-A
            # mean trajectory: buffer collapses to R_body when the worker
            # stops, and points forward when they are walking.
            hard_lobes, soft_lobes, buffer_curve = safety.inflate_all(
                ellipses, cfg.dt, belief, mean_traj=mean_traj,
            )

            # ---- Step A artists ----
            art["hist"].set_data(w["truth"][: frame + 1, 0],
                                 w["truth"][: frame + 1, 1])
            art["dot"].center = tuple(w["truth"][frame])
            art["tag"].set_position((float(w["truth"][frame, 0]),
                                     float(w["truth"][frame, 1]) + 0.42))
            art["mean"].set_data(mean_traj[:, 0], mean_traj[:, 1])
            for gi, line in enumerate(art["modes"]):
                line.set_data(mode_traj[gi, :, 0], mode_traj[gi, :, 1])
                line.set_alpha(float(np.clip(belief[gi] * 1.3, 0.05, 0.95)))
                line.set_linewidth(1.0 + 2.2 * float(belief[gi]))

            # ---- Step B swept safety tubes (convex hull of all teardrops) ----
            art["hard_tube"].set_xy(safety_tube_polygon(hard_lobes))
            art["soft_tube"].set_xy(safety_tube_polygon(soft_lobes))

            # ---- hard NO-GO teardrop at t=0 ----
            art["hard"].set_xy(hard_lobes[0])
            art["hard"].set_alpha(0.90)

            # ---- heading arrow showing which way the teardrop points ------
            if len(mean_traj) >= 2:
                d = mean_traj[1] - mean_traj[0]
                nrm = float(np.linalg.norm(d))
                if nrm > 1e-4:
                    head_dir = d / nrm
                    tail = tuple(w["truth"][frame])
                    tip = (tail[0] + float(head_dir[0]) * 0.85,
                           tail[1] + float(head_dir[1]) * 0.85)
                    art["heading"].set_position(tail)
                    art["heading"].xy = tip
                    art["heading"].arrow_patch.set_alpha(0.85)
                else:
                    art["heading"].arrow_patch.set_alpha(0.0)
            else:
                art["heading"].arrow_patch.set_alpha(0.0)

            # ---- KDE heatmap ----
            heat += _kde_heatmap(particles, grid_x, grid_y)

            # ---- intent bars ----
            for gi, (rect, txt) in enumerate(bar_groups[wi]):
                rect.set_width(bar_full_width * float(belief[gi]))
                txt.set_text(f"{belief[gi]:.2f}")
            entropy_texts[wi].set_text(f"H/Hmax={ent_norms[wi]:.2f}")

            # ---- safety-radius curve ----
            semi_max = np.maximum(ellipses[:, 2], ellipses[:, 3]) / 2.0
            r_total = semi_max + buffer_curve
            rad_lines[wi].set_data(horizon_times, r_total)

        if heat.max() > 1e-6:
            heat = heat / heat.max()
        heat_im.set_data(heat)

        # ---- inflation-breakdown bar (focus worker) ----
        focus_belief = worker_outs[focus_wi]["belief"]
        focus_mean = worker_outs[focus_wi]["mean"]
        if len(focus_mean) >= 2:
            v_at_T = float(np.linalg.norm(focus_mean[-1] - focus_mean[-2]) / cfg.dt)
        else:
            v_at_T = 0.0
        alpha_h = safety.velocity_gain(v_at_T, safety_cfg.v_full_walk)
        comps = safety.components(horizon_times[-1], focus_belief, v_human=v_at_T)
        left = 0.0
        for ki, name in enumerate(component_order):
            val = comps[name]
            rect = break_rects[ki]
            rect.set_xy((left, -0.20))
            rect.set_width(val)
            lbl = break_labels[ki]
            lbl.set_position((left + val * 0.5, 0.0))
            lbl.set_text(f"{val:.2f}")
            left += val
        break_subtitle.set_text(
            f"focus: {workers[focus_wi]['name']}   "
            f"H/Hmax={ent_norms[focus_wi]:.2f}   "
            f"v_h(T)={v_at_T:.2f} m/s   "
            f"alpha_h={alpha_h:.2f}   "
            f"R_buffer(T)={left:.2f} m"
        )

        r_max = safety.buffer_radius(
            horizon_times[-1], baseline_belief, v_human=safety_cfg.v_full_walk,
        )
        info_text.set_text(
            f"frame {frame+1}/{num_frames}    "
            f"horizon {cfg.horizon_steps * cfg.dt:.1f} s    "
            f"AMR v_max={safety_cfg.v_amr_max:.1f} m/s    "
            f"R_buffer range  [{safety.r_body:.2f}  ->  {r_max:.2f}] m  "
            f"(body-only  ->  walking, uniform intent)\n"
            "Step A:  recursive-Bayes intent  +  social-force rollout  +  95% ellipses  +  KDE\n"
            "Step B:  R_body + alpha_h(v_human) * (latency + braking + reaction*(1+H) + forecast*t)"
        )

        if not preview and (frame % 5 == 0 or frame == num_frames - 1):
            print(f"  rendering frame {frame+1}/{num_frames}")

        artists = [heat_im, info_text, break_subtitle]
        for art in w_artists:
            artists += [art["hist"], art["dot"], art["tag"], art["mean"]]
            artists += art["modes"]
            artists += [art["soft_tube"], art["hard_tube"], art["hard"]]
        for bars in bar_groups:
            for rect, txt in bars:
                artists += [rect, txt]
        artists += entropy_texts
        artists += rad_lines
        artists += break_rects + break_labels
        return artists

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

    if preview:
        print("Preview mode: showing live window. Close it to exit.")
        plt.show()
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(fps=10, bitrate=2400)
        anim.save(output_path, writer=writer)
        saved = output_path
    except Exception as exc:                                # pragma: no cover
        print(f"  ffmpeg unavailable ({exc}); falling back to GIF")
        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=8))
        saved = gif_path
    plt.close(fig)
    return saved
