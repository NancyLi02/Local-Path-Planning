"""Steps A->C demo animation (build_step_c_animation)."""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon, Rectangle

from ..tools.factory_map import GOALS, MAP_BOUNDS, OBSTACLES
from ..tools.prediction import PredictorConfig, IntentParticlePredictor
from ..tools.safety_inflation import SafetyInflationConfig, SafetyInflationModel
from ..tools.geometry import (
    safety_tube_polygon, point_in_polygon, polygon_signed_radius,
)
from ..tools.affected_amr import (
    AMR, CentralizedPlanner, ConflictChecker, ConflictResult,
    amr_human_collision, T_REPLAN_DEFAULT, AMR_SPEED_DEFAULT, V_AMR_TYPICAL_DEFAULT,
)
from ..tools.scenario import make_workers, make_amrs, make_stray_loader
from .render_common import (
    _rgba, _draw_factory, _kde_heatmap, _amr_body_polygon,
    _amr_tag_position, _ghost_border_color, STATUS_COLOR,
)


def build_step_c_animation(
    output_path: Path | None,
    num_frames: int = 280,
    num_workers: int = 2,
    num_amrs: int = 6,
    preview: bool = False,
    seed: int = 7,
    ghost_times_s: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    amr_safety_dist: float = 1.10,
    human_collision_dist: float = 0.55,
    t_replan: float = T_REPLAN_DEFAULT,
    v_amr_typical: float = V_AMR_TYPICAL_DEFAULT,
    inject_stray: bool = False,
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
    # Untracked humans -- no predictor, no safety tube. Collision check still
    # picks them up via direct geometric overlap.
    strays: list[dict] = []
    if inject_stray:
        strays.append(make_stray_loader(num_frames, cfg.dt))
    amrs = make_amrs(num_amrs)
    planner = CentralizedPlanner(
        amr_safety_dist=amr_safety_dist, dt=cfg.dt,
    )

    horizon_T = cfg.horizon_steps
    dt = cfg.dt
    horizon_seconds = horizon_T * dt
    ghost_idx = np.clip(
        np.array([int(round(t / dt)) - 1 for t in ghost_times_s]),
        0, horizon_T - 1,
    )

    # ----- figure layout -----
    fig = plt.figure(figsize=(16.0, 10.5), dpi=120)
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[3.0, 1.10],
        height_ratios=[11.5, 4.5, 0.7],
        wspace=0.18, hspace=0.32,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_status = fig.add_subplot(gs[0, 1])
    ax_gantt = fig.add_subplot(gs[1, :])
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    ax_status.axis("off")
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)

    _draw_factory(ax_map)
    ax_map.set_title(
        "Step C: Affected-AMR identification + centralized AMR-AMR yielding",
        fontsize=13, pad=10,
    )

    # ----- AMR rails (faint background lines) + QR markers -----
    for amr in amrs:
        ax_map.plot(amr.waypoints[:, 0], amr.waypoints[:, 1],
                    "-", color="#bdbdbd", lw=6.5, alpha=0.40,
                    solid_capstyle="round", zorder=1)
        ax_map.plot(amr.waypoints[:, 0], amr.waypoints[:, 1],
                    "--", color=amr.color, lw=1.1, alpha=0.55, zorder=1)
        # dense intermediate QR codes (~1 m apart) along the rail
        if len(amr.qr_points) > 2:
            ax_map.scatter(amr.qr_points[1:-1, 0], amr.qr_points[1:-1, 1],
                           marker="D", s=12, facecolor="white",
                           edgecolor="#8d8d8d", lw=0.6, alpha=0.85, zorder=2)
        # start QR marker
        p_start = amr.waypoints[0]
        ax_map.scatter(p_start[0], p_start[1], marker="s", s=44,
                       facecolor="white", edgecolor=amr.color, lw=1.6, zorder=2)
        # end QR marker
        p_end = amr.waypoints[-1]
        ax_map.scatter(p_end[0], p_end[1], marker="o", s=44,
                       facecolor=amr.color, edgecolor="white", lw=1.4, zorder=2)

    # ----- tracked worker handles (Step A predicted, Step B inflated) -----
    placeholder_poly = np.array([[0.0, 0.0], [0.0, 1e-3], [1e-3, 0.0]])
    worker_artists = []
    for w in workers:
        tube = Polygon(placeholder_poly, closed=True,
                       facecolor=_rgba(w["color"], 0.09),
                       edgecolor=_rgba(w["color"], 0.45),
                       lw=1.0, zorder=2)
        ax_map.add_patch(tube)
        hist_line, = ax_map.plot([], [], "-", lw=1.4,
                                 color=_rgba(w["color"], 0.50), zorder=3)
        dot = Circle((0, 0), 0.22, facecolor=w["color"],
                     edgecolor="white", lw=1.0, zorder=6)
        ax_map.add_patch(dot)
        tag = ax_map.text(
            0, 0, w["name"], fontsize=8.5, color="white",
            fontweight="bold", ha="center", va="bottom",
            bbox=dict(facecolor=w["color"], edgecolor="white",
                      lw=0.6, pad=2, boxstyle="round,pad=0.18"),
            zorder=7,
        )
        worker_artists.append(dict(tube=tube, hist=hist_line, dot=dot, tag=tag))

    # ----- stray (untracked) worker handles -- no tube, just a marked body
    stray_artists = []
    for s in strays:
        # Draw the full path lightly so the user sees the intruder ahead of time.
        ax_map.plot(s["truth"][:, 0], s["truth"][:, 1], ":",
                    lw=1.0, color=_rgba(s["color"], 0.45), zorder=1)
        hist_line, = ax_map.plot([], [], "-", lw=1.4,
                                 color=_rgba(s["color"], 0.55), zorder=3)
        dot = Circle((0, 0), 0.22, facecolor=s["color"],
                     edgecolor="white", lw=1.4, zorder=6,
                     hatch="//", alpha=0.0)
        ax_map.add_patch(dot)
        tag = ax_map.text(0, 0, f"{s['name']} (untracked)",
                          fontsize=7.5, color="white", fontweight="bold",
                          ha="center", va="bottom",
                          bbox=dict(facecolor=s["color"], edgecolor="white",
                                    lw=0.5, pad=1.4, boxstyle="round,pad=0.18"),
                          alpha=0.0, zorder=7)
        stray_artists.append(dict(hist=hist_line, dot=dot, tag=tag))

    # ----- AMR handles -----
    amr_artists = []
    for amr in amrs:
        body = Polygon(_amr_body_polygon(amr.waypoints[0, 0], amr.waypoints[0, 1], 0.0),
                       closed=True,
                       facecolor=amr.color, edgecolor="white", lw=1.5,
                       alpha=0.0, zorder=6)
        ax_map.add_patch(body)
        halo = Circle((0, 0), 0.70, facecolor="none",
                      edgecolor=STATUS_COLOR["CLEAR"], lw=2.5,
                      alpha=0.0, zorder=5)
        ax_map.add_patch(halo)
        planned_line, = ax_map.plot([], [], "-", lw=1.6,
                                    color=_rgba(amr.color, 0.85), zorder=4)
        ghosts = []
        for _ in ghost_times_s:
            g = Circle((0, 0), 0.20,
                       facecolor=amr.color,
                       edgecolor=STATUS_COLOR["CLEAR"], lw=1.8,
                       alpha=0.0, zorder=4)
            ax_map.add_patch(g)
            ghosts.append(g)
        conflict_line, = ax_map.plot([], [], "-", lw=1.6,
                                     color=STATUS_COLOR["REPLAN"],
                                     alpha=0.0, zorder=5)
        conflict_marker = Circle((0, 0), 0.30, facecolor="none",
                                 edgecolor=STATUS_COLOR["REPLAN"], lw=2.0,
                                 alpha=0.0, zorder=5)
        ax_map.add_patch(conflict_marker)
        # Collision explosion star (yellow filled, red outline)
        boom, = ax_map.plot([], [], marker="*", markersize=32,
                            markerfacecolor="#fff176",
                            markeredgecolor=STATUS_COLOR["COLLISION"],
                            markeredgewidth=2.2, linestyle="None",
                            alpha=0.0, zorder=7)
        boom_text = ax_map.text(0, 0, "BOOM", fontsize=8,
                                color=STATUS_COLOR["COLLISION"],
                                fontweight="bold", ha="center", va="bottom",
                                alpha=0.0, zorder=7)
        tag = ax_map.text(
            0, 0, amr.name, fontsize=8.5, color="white",
            fontweight="bold", ha="center", va="center",
            bbox=dict(facecolor=amr.color, edgecolor="white",
                      lw=0.6, pad=2, boxstyle="round,pad=0.18"),
            alpha=0.0, zorder=8,
        )
        amr_artists.append(dict(
            body=body, halo=halo, planned=planned_line,
            ghosts=ghosts, conf_line=conflict_line, conf_mark=conflict_marker,
            boom=boom, boom_text=boom_text, tag=tag,
        ))

    info_text = ax_map.text(
        MAP_BOUNDS[0] + 0.4, MAP_BOUNDS[3] - 0.3, "",
        fontsize=9, color="#222",
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        zorder=8, va="top", ha="left",
    )

    # ----- right-side per-AMR status cards (6 compact cards) -----
    n_cards = len(amrs)
    ax_status.text(
        0.5, 0.995, "Fleet status",
        fontsize=11.5, fontweight="bold", color="#222",
        ha="center", va="top", transform=ax_status.transAxes,
    )
    top_margin = 0.045
    bot_margin = 0.020
    gap = 0.013
    card_height = (1.0 - top_margin - bot_margin - gap * (n_cards - 1)) / n_cards
    card_handles = []
    for i, amr in enumerate(amrs):
        top = 1.0 - top_margin - i * (card_height + gap)
        bot = top - card_height
        bg = FancyBboxPatch(
            (0.02, bot + 0.003), 0.96, card_height - 0.006,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor="white", edgecolor=amr.color, lw=1.5,
            transform=ax_status.transAxes,
        )
        ax_status.add_patch(bg)
        # identity dot + name
        y_name = top - card_height * 0.22
        ax_status.text(0.05, y_name, "\u25cf",
                       fontsize=14, color=amr.color,
                       transform=ax_status.transAxes, va="center")
        name_txt = ax_status.text(0.10, y_name, amr.name,
                                  fontsize=10.5, fontweight="bold", color="#222",
                                  transform=ax_status.transAxes, va="center")
        speed_txt = ax_status.text(
            0.97, y_name, f"v={amr.commanded_speed:.2f}",
            fontsize=8.5, color="#444",
            transform=ax_status.transAxes, va="center", ha="right",
        )
        # status badge
        y_badge = top - card_height * 0.56
        badge_w = 0.30
        badge_x = 0.05
        badge = FancyBboxPatch(
            (badge_x, y_badge - card_height * 0.17),
            badge_w, card_height * 0.32,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            facecolor="#cccccc", edgecolor="none",
            transform=ax_status.transAxes,
        )
        ax_status.add_patch(badge)
        badge_text = ax_status.text(
            badge_x + badge_w * 0.5, y_badge, "—",
            fontsize=9.0, color="white", fontweight="bold",
            ha="center", va="center",
            transform=ax_status.transAxes,
        )
        # detail line (right of badge)
        detail_text = ax_status.text(
            badge_x + badge_w + 0.025, y_badge, "",
            fontsize=8.7, color="#222",
            transform=ax_status.transAxes, va="center",
        )
        # secondary line (below badge) for the cause / partner
        y_sub = top - card_height * 0.86
        sub_text = ax_status.text(
            0.06, y_sub, "",
            fontsize=8.0, color="#555",
            transform=ax_status.transAxes, va="center",
        )
        card_handles.append(dict(
            badge=badge, badge_text=badge_text,
            detail=detail_text, sub=sub_text,
            name=name_txt, speed=speed_txt,
            amr=amr,
        ))

    # ----- bottom Gantt chart -----
    ax_gantt.set_xlim(0, horizon_seconds)
    ax_gantt.set_ylim(-0.5, n_cards - 0.5)
    ax_gantt.invert_yaxis()
    ax_gantt.set_yticks(range(n_cards))
    ax_gantt.set_yticklabels([amr.name for amr in amrs], fontsize=9)
    ax_gantt.set_xlabel("Lookahead horizon t [s]")
    ax_gantt.set_title(
        f"Per-AMR space-time conflict status "
        f"(red zone < t_replan_eff = α_amr(v)·{t_replan:.1f}s  →  trigger Step D; "
        "α_amr=0 ⇒ stationary AMR never REPLANs)",
        fontsize=10.5, pad=6,
    )
    ax_gantt.set_facecolor("#fafafa")
    ax_gantt.grid(True, alpha=0.30, axis="x")
    for spine in ("top", "right"):
        ax_gantt.spines[spine].set_visible(False)
    ax_gantt.axvline(0.0, color="#444", lw=1.0)
    ax_gantt.axvline(horizon_seconds, color="#444", lw=1.0, linestyle="--", alpha=0.7)
    # REPLAN threshold line: any hard cell to its LEFT promotes the AMR to REPLAN.
    ax_gantt.axvline(t_replan, color=STATUS_COLOR["REPLAN"], lw=1.8, linestyle="--",
                     alpha=0.8)
    ax_gantt.text(
        t_replan - 0.05, -0.55, f"t_replan = {t_replan:.1f}s",
        fontsize=8.5, color=STATUS_COLOR["REPLAN"], ha="right", va="top",
        fontweight="bold",
    )

    gantt_cells = []
    gantt_collision_stars = []          # black star at t=0 of a collided row
    gantt_collision_labels = []         # right-side text annotation
    gantt_collision_overlays = []       # black hatched overlay across the row
    for ai in range(n_cards):
        row_cells = []
        for t in range(horizon_T):
            rect = Rectangle((t * dt, ai - 0.36), dt, 0.72,
                             facecolor=STATUS_COLOR["PENDING"],
                             edgecolor="white", lw=0.4)
            ax_gantt.add_patch(rect)
            row_cells.append(rect)
        gantt_cells.append(row_cells)

        # Diagonal black hatch overlay — drawn on top of the frozen pattern so
        # the user can still read the underlying conflict cells but immediately
        # recognises the row as a collided AMR. Hidden until collision.
        overlay = Rectangle(
            (0.0, ai - 0.36), horizon_seconds, 0.72,
            facecolor="none",
            edgecolor=STATUS_COLOR["COLLISION"],
            hatch="//", linewidth=0.0,
            alpha=0.0, zorder=4,
        )
        ax_gantt.add_patch(overlay)
        gantt_collision_overlays.append(overlay)

        # Big black star pinned at t=0 of the row.
        star, = ax_gantt.plot(
            [], [], marker="*", markersize=18,
            markerfacecolor=STATUS_COLOR["COLLISION"],
            markeredgecolor="white", markeredgewidth=1.4,
            linestyle="None", zorder=6, alpha=0.0,
        )
        gantt_collision_stars.append(star)

        # Right-side label "COLLISION @ frame N (Tech-1)".
        label = ax_gantt.text(
            horizon_seconds * 0.97, ai, "",
            fontsize=8.5, color="white", fontweight="bold",
            ha="right", va="center", alpha=0.0, zorder=7,
            bbox=dict(facecolor=STATUS_COLOR["COLLISION"],
                      edgecolor="white", linewidth=1.0,
                      boxstyle="round,pad=0.22"),
        )
        gantt_collision_labels.append(label)

    # ----- legend -----
    legend_handles = [
        plt.Line2D([0], [0], color="#bdbdbd", lw=6.5, alpha=0.65),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor="white", markeredgecolor="#888",
                   markersize=8, linestyle="None"),
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="white", markeredgecolor="#8d8d8d",
                   markersize=6, linestyle="None"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#888", markeredgecolor="white",
                   markersize=9, linestyle="None"),
        Patch(facecolor=_rgba("#888888", 0.10), edgecolor="#888"),
        Patch(facecolor="#888", edgecolor=STATUS_COLOR["CLEAR"], linewidth=2.0),
        Patch(facecolor="#888", edgecolor=STATUS_COLOR["WATCH"], linewidth=2.0),
        Patch(facecolor="#888", edgecolor=STATUS_COLOR["SLOWDOWN"], linewidth=2.0),
        Patch(facecolor="#888", edgecolor=STATUS_COLOR["REPLAN"], linewidth=2.0),
        plt.Line2D([0], [0], color=STATUS_COLOR["REPLAN"], lw=1.8),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#fff176",
                   markeredgecolor=STATUS_COLOR["COLLISION"],
                   markersize=18, linestyle="None"),
        Patch(facecolor="white", edgecolor=STATUS_COLOR["COLLISION"],
              hatch="//", linewidth=1.0),
    ]
    legend_labels = [
        "AMR reference rail",
        "QR start",
        "QR waypoint (dense floor grid)",
        "QR goal",
        "Tracked worker safety tube (Step B)",
        "Ghost border: CLEAR",
        "Ghost border: WATCH (soft hit, or stationary AMR)",
        "Ghost border: SLOWDOWN (hard, TTC ≥ t_replan_eff)",
        "Ghost border: REPLAN (hard, TTC < t_replan_eff)",
        "REPLAN connector (ghost @ TTC → worker)",
        "Physical AMR-human collision (AMR disabled)",
        "Gantt row: collided AMR (hatched overlay + ★)",
    ]
    if strays:
        # When a stray is injected, surface it explicitly in the legend.
        legend_handles.insert(4,
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#d84315", markeredgecolor="white",
                       markersize=10, linestyle="None"))
        legend_labels.insert(4, "Untracked human (Loader, no tube)")
    ax_leg.legend(legend_handles, legend_labels, loc="center", ncol=4,
                  frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=8.0)

    # ----- per-frame update -----
    def update(frame: int):
        # 1a) Tracked workers + Step-A predictions + Step-B teardrop lobes
        worker_data = []
        for w, predictor, art in zip(workers, predictors, worker_artists):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            out = predictor.rollout(obs)
            ellipses = out["ellipses"]
            # Pass mean_traj so the buffer auto-shrinks when the worker
            # stops (alpha_h gating) AND points forward when they walk
            # (heading -> teardrop orientation).
            hard_lobes, soft_lobes, _ = safety.inflate_all(
                ellipses, dt, out["belief"], mean_traj=out["mean"],
            )
            centers = ellipses[:, :2]

            hull = safety_tube_polygon(hard_lobes)
            art["tube"].set_xy(hull)
            art["hist"].set_data(w["truth"][: frame + 1, 0],
                                 w["truth"][: frame + 1, 1])
            art["dot"].center = tuple(w["truth"][frame])
            art["tag"].set_position((float(w["truth"][frame, 0]),
                                     float(w["truth"][frame, 1]) + 0.42))

            worker_data.append(dict(
                name=w["name"], color=w["color"],
                inflated=hard_lobes, soft=soft_lobes,
                centers=centers,
            ))

        # 1b) Stray (untracked) humans -- no prediction, no tube. Only their
        # ground-truth position participates, via the collision check.
        all_humans = list(workers)
        for s, sart in zip(strays, stray_artists):
            spawned = frame >= s.get("spawn_frame", 0)
            pos = s["truth"][frame] if frame < len(s["truth"]) else s["truth"][-1]
            sart["dot"].center = tuple(pos)
            sart["dot"].set_alpha(1.0 if spawned else 0.0)
            sart["hist"].set_data(s["truth"][: frame + 1, 0],
                                  s["truth"][: frame + 1, 1])
            sart["tag"].set_position((pos[0], pos[1] + 0.6))
            sart["tag"].set_alpha(1.0 if spawned else 0.0)
            all_humans.append(s)

        # 2) Centralized planner sets actual_speed for all active AMRs
        # (collided AMRs are filtered out via is_active and stay frozen)
        planner.resolve(amrs, frame)

        # 3) Per-AMR conflict + collision + status + render
        any_collision_now = False
        any_replan = 0
        any_slowdown = 0
        any_waiting = 0
        for ai, (amr, art, card) in enumerate(zip(amrs, amr_artists, card_handles)):
            spawned = amr.is_spawned(frame)
            done = amr.is_done()

            # ------------------------------------------------------------
            # Inactive (PENDING / DONE / COLLIDED-but-not-yet-rendered)
            # ------------------------------------------------------------
            if amr.collided:
                # Persistent frozen rendering at collision pose.
                cx, cy = amr.collision_pos
                head = amr.collision_heading
                art["body"].set_xy(_amr_body_polygon(cx, cy, head))
                art["body"].set_facecolor(amr.color)
                art["body"].set_alpha(0.45)            # desaturated to read "disabled"
                art["halo"].center = (cx, cy)
                art["halo"].set_edgecolor(STATUS_COLOR["COLLISION"])
                art["halo"].set_alpha(0.95)
                art["halo"].set_linewidth(3.2)
                art["planned"].set_data([], [])
                for g in art["ghosts"]:
                    g.set_alpha(0.0)
                art["conf_line"].set_alpha(0.0)
                art["conf_mark"].set_alpha(0.0)
                art["boom"].set_data([cx], [cy])
                art["boom"].set_alpha(1.0)
                art["boom_text"].set_position((cx, cy + 0.55))
                art["boom_text"].set_alpha(1.0)
                art["boom_text"].set_text("DISABLED")
                ltx, lty = _amr_tag_position(cx, cy, head)
                art["tag"].set_position((ltx, lty))
                art["tag"].set_alpha(0.85)

                # Gantt row is FROZEN at the conflict pattern captured at the
                # moment of collision -- it does not advance any more. We just
                # re-stamp it every frame so animation blitting can pick it up.
                hard_mask = amr.frozen_hard_mask
                soft_mask = amr.frozen_soft_mask
                t_replan_eff_f = amr.frozen_t_replan_eff
                alpha_amr_f = amr.frozen_alpha_amr
                for t in range(horizon_T):
                    t_val = (t + 1) * dt
                    if hard_mask is not None and bool(hard_mask[t]):
                        if alpha_amr_f < 0.05:
                            color = STATUS_COLOR["WATCH"]
                        elif t_val < t_replan_eff_f:
                            color = STATUS_COLOR["REPLAN"]
                        else:
                            color = STATUS_COLOR["SLOWDOWN"]
                    elif soft_mask is not None and bool(soft_mask[t]):
                        color = STATUS_COLOR["WATCH"]
                    else:
                        color = STATUS_COLOR["CLEAR"]
                    gantt_cells[ai][t].set_facecolor(color)

                # Persistent collision indicators on the gantt row.
                gantt_collision_overlays[ai].set_alpha(0.55)
                gantt_collision_stars[ai].set_data([0.06], [ai])
                gantt_collision_stars[ai].set_alpha(1.0)
                gantt_collision_labels[ai].set_text(
                    f"COLLISION @ f{amr.collision_frame}  ({amr.collision_worker})"
                )
                gantt_collision_labels[ai].set_alpha(1.0)

                card["badge"].set_facecolor(STATUS_COLOR["COLLISION"])
                card["badge_text"].set_text("COLLISION")
                card["detail"].set_text(f"hit {amr.collision_worker} @ frame {amr.collision_frame}")
                card["sub"].set_text("AMR frozen / gantt locked at impact")
                card["speed"].set_text("v=0.00")
                continue

            if not spawned or done:
                art["body"].set_alpha(0.0)
                art["halo"].set_alpha(0.0)
                art["planned"].set_data([], [])
                for g in art["ghosts"]:
                    g.set_alpha(0.0)
                art["conf_line"].set_alpha(0.0)
                art["conf_mark"].set_alpha(0.0)
                art["boom"].set_alpha(0.0)
                art["boom_text"].set_alpha(0.0)
                art["tag"].set_alpha(0.0)

                if not spawned:
                    status = "PENDING"
                    detail = f"spawns @ frame {amr.spawn_frame}"
                else:
                    status = "DONE"
                    detail = "path completed"

                for t in range(horizon_T):
                    gantt_cells[ai][t].set_facecolor(STATUS_COLOR[status])

                # No collision on this row.
                gantt_collision_overlays[ai].set_alpha(0.0)
                gantt_collision_stars[ai].set_alpha(0.0)
                gantt_collision_labels[ai].set_alpha(0.0)

                card["badge"].set_facecolor(STATUS_COLOR[status])
                card["badge_text"].set_text(status)
                card["detail"].set_text(detail)
                card["sub"].set_text("")
                card["speed"].set_text("v=—")
                continue

            # ------------------------------------------------------------
            # Active AMR: check for new physical collision FIRST
            # ------------------------------------------------------------
            collided_now, hit_worker = amr_human_collision(
                amr, frame, all_humans, collision_dist=human_collision_dist,
            )
            if collided_now:
                # Run the Step-C check once so we can SNAPSHOT the conflict
                # pattern at the instant of impact -- this becomes the frozen
                # gantt row for the rest of the simulation.
                impact_result = ConflictChecker.check(
                    amr, dt, horizon_T, worker_data,
                    t_replan=t_replan, v_amr_typical=v_amr_typical,
                )
                amr.mark_collision(
                    frame, hit_worker,
                    hard_mask=impact_result.hard_mask,
                    soft_mask=impact_result.soft_mask,
                    alpha_amr=impact_result.alpha_amr,
                    t_replan_eff=impact_result.t_replan_eff,
                )
                any_collision_now = True

                cx, cy = amr.collision_pos
                head = amr.collision_heading
                art["body"].set_xy(_amr_body_polygon(cx, cy, head))
                art["body"].set_facecolor(amr.color)
                art["body"].set_alpha(0.55)
                art["halo"].center = (cx, cy)
                art["halo"].set_edgecolor(STATUS_COLOR["COLLISION"])
                art["halo"].set_alpha(1.0)
                art["halo"].set_linewidth(3.5)
                art["planned"].set_data([], [])
                for g in art["ghosts"]:
                    g.set_alpha(0.0)
                art["conf_line"].set_alpha(0.0)
                art["conf_mark"].set_alpha(0.0)
                art["boom"].set_data([cx], [cy])
                art["boom"].set_alpha(1.0)
                art["boom_text"].set_position((cx, cy + 0.55))
                art["boom_text"].set_alpha(1.0)
                art["boom_text"].set_text(f"BOOM  ({hit_worker})")
                ltx, lty = _amr_tag_position(cx, cy, head)
                art["tag"].set_position((ltx, lty))
                art["tag"].set_alpha(0.85)

                # Lock in the gantt row at the impact-instant conflict pattern.
                alpha_amr_f = impact_result.alpha_amr
                t_replan_eff_f = impact_result.t_replan_eff
                for t in range(horizon_T):
                    t_val = (t + 1) * dt
                    if bool(impact_result.hard_mask[t]):
                        if alpha_amr_f < 0.05:
                            color = STATUS_COLOR["WATCH"]
                        elif t_val < t_replan_eff_f:
                            color = STATUS_COLOR["REPLAN"]
                        else:
                            color = STATUS_COLOR["SLOWDOWN"]
                    elif bool(impact_result.soft_mask[t]):
                        color = STATUS_COLOR["WATCH"]
                    else:
                        color = STATUS_COLOR["CLEAR"]
                    gantt_cells[ai][t].set_facecolor(color)

                # Persistent collision indicators on the gantt row.
                gantt_collision_overlays[ai].set_alpha(0.55)
                gantt_collision_stars[ai].set_data([0.06], [ai])
                gantt_collision_stars[ai].set_alpha(1.0)
                gantt_collision_labels[ai].set_text(
                    f"COLLISION @ f{amr.collision_frame}  ({amr.collision_worker})"
                )
                gantt_collision_labels[ai].set_alpha(1.0)

                card["badge"].set_facecolor(STATUS_COLOR["COLLISION"])
                card["badge_text"].set_text("COLLISION")
                card["detail"].set_text(f"hit  {hit_worker}")
                card["sub"].set_text("AMR frozen / gantt locked at impact")
                card["speed"].set_text("v=0.00")
                continue

            # Step-C check uses ACTUAL speed (WAITING -> stationary projection)
            # The checker also velocity-gates REPLAN against amr.actual_speed,
            # so a stationary AMR (alpha_amr ~= 0) can never escalate beyond
            # WATCH no matter how close the worker tube comes.
            result = ConflictChecker.check(
                amr, dt, horizon_T, worker_data,
                t_replan=t_replan, v_amr_typical=v_amr_typical,
            )

            # Status priority:  WAITING > REPLAN > SLOWDOWN > WATCH > CLEAR
            # (WAITING dominates because the centralised planner has already
            # acted; Step-C info appears in the sub-line.)
            t_replan_eff = result.t_replan_eff
            if amr.waiting_for:
                status = "WAITING"
                detail = (f"hold @ QR#{amr.current_qr_index()}  "
                          f"(cell held by {amr.waiting_for})")
                if result.status == "REPLAN":
                    sub = f"Step-C: REPLAN, {result.closest_worker} @ {result.ttc:.1f}s"
                elif result.status == "SLOWDOWN":
                    sub = f"Step-C: SLOWDOWN, {result.closest_worker} @ {result.ttc:.1f}s"
                elif result.status == "WATCH":
                    sub = f"Step-C: WATCH, near {result.closest_worker}"
                else:
                    sub = "Step-C: clear"
                any_waiting += 1
            elif result.status == "REPLAN":
                status = "REPLAN"
                detail = f"TTC = {result.ttc:.1f}s  <  {t_replan_eff:.2f}s"
                sub = f"trigger Step D  ({result.closest_worker})"
                any_replan += 1
            elif result.status == "SLOWDOWN":
                status = "SLOWDOWN"
                detail = f"TTC = {result.ttc:.1f}s"
                sub = f"decelerate, monitor  ({result.closest_worker})"
                any_slowdown += 1
            elif result.status == "WATCH":
                status = "WATCH"
                if result.hard_mask.any() and result.alpha_amr < 0.05:
                    detail = "stationary -- no replan"
                    sub = f"hold ground  ({result.closest_worker})"
                else:
                    detail = f"soft margin {result.margin:.2f}"
                    sub = f"observe  ({result.closest_worker})"
            else:
                status = "CLEAR"
                detail = f"margin {result.margin:.2f}"
                sub = "no conflict in 5 s"

            status_clr = STATUS_COLOR[status]

            # Body + halo
            cx, cy = amr.position_at(amr.progress)
            head = amr.heading_at(amr.progress)
            art["body"].set_xy(_amr_body_polygon(cx, cy, head))
            art["body"].set_facecolor(amr.color)
            art["body"].set_alpha(1.0)
            art["halo"].center = (cx, cy)
            art["halo"].set_edgecolor(status_clr)
            if status == "REPLAN":
                art["halo"].set_alpha(0.95)
                art["halo"].set_linewidth(3.0)
            elif status in ("WAITING", "SLOWDOWN"):
                art["halo"].set_alpha(0.80)
                art["halo"].set_linewidth(2.3)
            elif status == "WATCH":
                art["halo"].set_alpha(0.55)
                art["halo"].set_linewidth(1.8)
            else:
                art["halo"].set_alpha(0.20)
                art["halo"].set_linewidth(1.3)

            # Planned 5 s trajectory
            art["planned"].set_data(
                result.pred_positions[:, 0], result.pred_positions[:, 1],
            )

            # Future ghosts -- fill = AMR identity, border = per-t status color
            for k, t_idx in enumerate(ghost_idx):
                gpos = result.pred_positions[t_idx]
                t_val = (t_idx + 1) * dt
                g = art["ghosts"][k]
                g.center = (float(gpos[0]), float(gpos[1]))
                g.set_facecolor(_rgba(amr.color, 0.88 - 0.10 * k))
                g.set_edgecolor(_ghost_border_color(
                    bool(result.hard_mask[t_idx]),
                    bool(result.soft_mask[t_idx]),
                    t_val, t_replan_eff, result.alpha_amr,
                ))
                g.set_radius(max(0.23 - 0.025 * k, 0.10))
                g.set_alpha(0.92 - 0.08 * k)

            # REPLAN connector: only when the first hard-tube hit is inside t_replan
            if result.status == "REPLAN" and result.closest_worker_pos is not None:
                t_star = int(np.argmax(result.hard_mask))
                amr_at_t = result.pred_positions[t_star]
                wpos = result.closest_worker_pos
                art["conf_line"].set_data(
                    [amr_at_t[0], wpos[0]], [amr_at_t[1], wpos[1]],
                )
                art["conf_line"].set_alpha(0.90)
                art["conf_mark"].center = (float(amr_at_t[0]), float(amr_at_t[1]))
                art["conf_mark"].set_alpha(0.90)
            else:
                art["conf_line"].set_alpha(0.0)
                art["conf_mark"].set_alpha(0.0)

            # Boom is reserved for actual COLLISION rendering only
            art["boom"].set_alpha(0.0)
            art["boom_text"].set_alpha(0.0)

            # Name badge follows the AMR body.
            ltx, lty = _amr_tag_position(cx, cy, head)
            art["tag"].set_position((ltx, lty))
            art["tag"].set_alpha(1.0)

            # Card
            card["badge"].set_facecolor(status_clr)
            card["badge_text"].set_text(status)
            card["detail"].set_text(detail)
            card["sub"].set_text(sub)
            card["speed"].set_text(f"v={amr.actual_speed:.2f}")

            # Gantt row colors: per-cell time-slice severity. The cells use
            # the *effective* REPLAN window (which shrinks with AMR speed):
            # for a stationary AMR every hit collapses to WATCH because no
            # replanning is actionable.
            for t in range(horizon_T):
                t_val = (t + 1) * dt
                if result.hard_mask[t]:
                    if result.alpha_amr < 0.05:
                        color = STATUS_COLOR["WATCH"]
                    elif t_val < t_replan_eff:
                        color = STATUS_COLOR["REPLAN"]
                    else:
                        color = STATUS_COLOR["SLOWDOWN"]
                elif result.soft_mask[t]:
                    color = STATUS_COLOR["WATCH"]
                else:
                    color = STATUS_COLOR["CLEAR"]
                gantt_cells[ai][t].set_facecolor(color)

            # No collision on this row.
            gantt_collision_overlays[ai].set_alpha(0.0)
            gantt_collision_stars[ai].set_alpha(0.0)
            gantt_collision_labels[ai].set_alpha(0.0)

        # 4) Advance AMRs (after status snapshot for this frame is captured).
        # is_active already filters out collided / pending / done AMRs.
        for amr in amrs:
            if amr.is_active(frame):
                amr.step(dt)

        # Summary info
        n_active = sum(1 for a in amrs if a.is_active(frame))
        n_pending = sum(1 for a in amrs if not a.is_spawned(frame))
        n_done = sum(1 for a in amrs if a.is_done())
        n_collided = sum(1 for a in amrs if a.collided)
        info_text.set_text(
            f"frame {frame+1}/{num_frames}    "
            f"active {n_active} / pending {n_pending} / done {n_done} / collided {n_collided}    "
            f"REPLAN {any_replan}  SLOWDOWN {any_slowdown}  WAITING {any_waiting}"
            + ("    NEW COLLISION!" if any_collision_now else "")
            + f"\nCentralizedPlanner: QR-cell reservation, FCFS by waypoint-arrival   "
            f"Step-D trigger:  REPLAN  (TTC < t_replan = {t_replan:.1f}s)"
        )

        if not preview and (frame % 5 == 0 or frame == num_frames - 1):
            print(f"  rendering frame {frame+1}/{num_frames}")

        artists = [info_text]
        for art in worker_artists:
            artists += [art["tube"], art["hist"], art["dot"], art["tag"]]
        for sart in stray_artists:
            artists += [sart["hist"], sart["dot"], sart["tag"]]
        for art in amr_artists:
            artists += [art["body"], art["halo"], art["planned"],
                        art["conf_line"], art["conf_mark"],
                        art["boom"], art["boom_text"], art["tag"]]
            artists += art["ghosts"]
        for card in card_handles:
            artists += [card["badge"], card["badge_text"],
                        card["detail"], card["sub"], card["speed"]]
        for row in gantt_cells:
            artists += row
        artists += gantt_collision_overlays
        artists += gantt_collision_stars
        artists += gantt_collision_labels
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
