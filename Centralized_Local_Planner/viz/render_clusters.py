"""Steps A->D demo animation (build_step_d_animation)."""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import (
    Circle, FancyBboxPatch, FancyArrowPatch, Patch, Polygon, Rectangle,
)

from ..tools.factory_map import GOALS, MAP_BOUNDS, OBSTACLES
from ..tools.prediction import PredictorConfig, IntentParticlePredictor
from ..tools.safety_inflation import SafetyInflationConfig, SafetyInflationModel
from ..tools.geometry import (
    safety_tube_polygon, point_in_polygon, polygon_signed_radius,
    _convex_hull, _expand_hull,
)
from ..tools.affected_amr import (
    AMR, CentralizedPlanner, ConflictChecker, ConflictResult,
    amr_human_collision, T_REPLAN_DEFAULT, AMR_SPEED_DEFAULT, V_AMR_TYPICAL_DEFAULT,
)
from ..tools.conflict_cluster import (
    ConflictCluster, ClusterResult, ConflictClusterBuilder,
    CLUSTER_PALETTE, SINGLETON_COLOR,
)
from ..tools.scenario import make_workers, make_amrs
from .render_common import (
    _rgba, _draw_factory, _kde_heatmap, _amr_body_polygon,
    _amr_tag_position, _ghost_border_color, STATUS_COLOR,
)


_EDGE_COLOR = "#ff5722"
_EDGE_ALPHA = 0.85
_EDGE_LW = 2.2

def build_step_d_animation(
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
    cluster_spatial_dist: float = 2.5,
    cascade_dist: float = 2.0,
    replan_region_buffer: float = 1.8,
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
    amrs = make_amrs(num_amrs)
    planner = CentralizedPlanner(amr_safety_dist=amr_safety_dist, dt=cfg.dt)
    cluster_builder = ConflictClusterBuilder(
        cluster_spatial_dist=cluster_spatial_dist,
        cascade_dist=cascade_dist,
        replan_region_buffer=replan_region_buffer,
    )

    dt = cfg.dt
    horizon_T = cfg.horizon_steps
    horizon_seconds = horizon_T * dt
    ghost_idx = np.clip(
        np.array([int(round(t / dt)) - 1 for t in ghost_times_s]),
        0, horizon_T - 1,
    )

    # ------------------------------------------------------------------ layout
    # 3-panel layout:
    #   [map (wide) | cluster cards (narrow)]
    #   [dist matrix heat-map (full width)]
    #   [legend (full width)]
    fig = plt.figure(figsize=(17.5, 11.5), dpi=120)
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[3.2, 1.1],
        height_ratios=[10.5, 4.2, 0.75],
        wspace=0.17, hspace=0.36,
    )
    ax_map    = fig.add_subplot(gs[0, 0])
    ax_cards  = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[1, :])
    ax_leg    = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)

    _draw_factory(ax_map)
    ax_map.set_title(
        "Step D: Conflict Cluster Construction  +  Local Replanning Region",
        fontsize=13, pad=10,
    )

    # ------------------------------------------------------------------ AMR rails
    for amr in amrs:
        ax_map.plot(amr.waypoints[:, 0], amr.waypoints[:, 1],
                    "-", color="#bdbdbd", lw=6.0, alpha=0.38,
                    solid_capstyle="round", zorder=1)
        ax_map.plot(amr.waypoints[:, 0], amr.waypoints[:, 1],
                    "--", color=amr.color, lw=1.0, alpha=0.50, zorder=1)
        if len(amr.qr_points) > 2:
            ax_map.scatter(amr.qr_points[1:-1, 0], amr.qr_points[1:-1, 1],
                           marker="D", s=10, facecolor="white",
                           edgecolor="#aaaaaa", lw=0.5, alpha=0.75, zorder=2)
        ax_map.scatter(*amr.waypoints[0],  marker="s", s=40,
                       facecolor="white", edgecolor=amr.color, lw=1.5, zorder=2)
        ax_map.scatter(*amr.waypoints[-1], marker="o", s=40,
                       facecolor=amr.color, edgecolor="white", lw=1.3, zorder=2)

    # ------------------------------------------------------------------ worker artists
    placeholder = np.array([[0.0, 0.0], [0.0, 1e-3], [1e-3, 0.0]])
    worker_artists = []
    for w in workers:
        tube = Polygon(placeholder, closed=True,
                       facecolor=_rgba(w["color"], 0.08),
                       edgecolor=_rgba(w["color"], 0.40),
                       lw=1.0, zorder=2)
        ax_map.add_patch(tube)
        hist, = ax_map.plot([], [], "-", lw=1.3,
                            color=_rgba(w["color"], 0.45), zorder=3)
        dot = Circle((0, 0), 0.22, facecolor=w["color"],
                     edgecolor="white", lw=1.0, zorder=6)
        ax_map.add_patch(dot)
        tag = ax_map.text(0, 0, w["name"], fontsize=8, color="white",
                          fontweight="bold", ha="center", va="bottom",
                          bbox=dict(facecolor=w["color"], edgecolor="white",
                                    lw=0.5, pad=1.8, boxstyle="round,pad=0.16"),
                          zorder=7)
        worker_artists.append(dict(tube=tube, hist=hist, dot=dot, tag=tag))

    # ------------------------------------------------------------------ AMR artists
    amr_artists = []
    for amr in amrs:
        body = Polygon(_amr_body_polygon(*amr.waypoints[0], 0.0),
                       closed=True,
                       facecolor=amr.color, edgecolor="white", lw=1.4,
                       alpha=0.0, zorder=6)
        ax_map.add_patch(body)
        halo = Circle((0, 0), 0.68, facecolor="none",
                      edgecolor=STATUS_COLOR["CLEAR"], lw=2.3,
                      alpha=0.0, zorder=5)
        ax_map.add_patch(halo)
        planned, = ax_map.plot([], [], "-", lw=1.4,
                               color=_rgba(amr.color, 0.80), zorder=4)
        ghosts = []
        for _ in ghost_times_s:
            g = Circle((0, 0), 0.20, facecolor=amr.color,
                       edgecolor=STATUS_COLOR["CLEAR"], lw=1.7,
                       alpha=0.0, zorder=4)
            ax_map.add_patch(g)
            ghosts.append(g)
        tag = ax_map.text(0, 0, amr.name, fontsize=8, color="white",
                          fontweight="bold", ha="center", va="center",
                          bbox=dict(facecolor=amr.color, edgecolor="white",
                                    lw=0.5, pad=1.8, boxstyle="round,pad=0.16"),
                          alpha=0.0, zorder=8)
        amr_artists.append(dict(body=body, halo=halo,
                                planned=planned, ghosts=ghosts, tag=tag))

    # ------------------------------------------------------------------ cluster overlays
    # Pre-allocate enough cluster hull polygons (max = num_amrs // 2).
    max_clusters = max(num_amrs // 2, 3)
    cluster_hulls = []
    cluster_labels = []
    for _ in range(max_clusters):
        ph = Polygon(placeholder, closed=True, facecolor="none",
                     edgecolor="#cccccc", lw=0.0, alpha=0.0, zorder=3)
        ax_map.add_patch(ph)
        cluster_hulls.append(ph)
        lbl = ax_map.text(0, 0, "", fontsize=9, fontweight="bold",
                          ha="center", va="center", color="white",
                          bbox=dict(facecolor="#cccccc", edgecolor="white",
                                    lw=0.6, pad=2, boxstyle="round,pad=0.20"),
                          alpha=0.0, zorder=9)
        cluster_labels.append(lbl)

    # Coupling edges (pre-allocate: at most N*(N-1)/2 lines).
    max_edges = num_amrs * (num_amrs - 1) // 2
    coupling_lines = [
        ax_map.plot([], [], "-", lw=_EDGE_LW, color=_EDGE_COLOR,
                    alpha=0.0, zorder=5)[0]
        for _ in range(max_edges)
    ]

    info_text = ax_map.text(
        MAP_BOUNDS[0] + 0.4, MAP_BOUNDS[3] - 0.3, "",
        fontsize=9, color="#222",
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        zorder=10, va="top", ha="left",
    )

    # ------------------------------------------------------------------ cluster cards panel
    ax_cards.set_title("Conflict Clusters", fontsize=11.5,
                       fontweight="bold", pad=8)
    # We draw cards dynamically every frame (text artists updated in place).
    card_bg_patches: list[FancyBboxPatch] = []
    card_texts: list[dict] = []
    _MAX_CARDS = max_clusters + 2           # +2 for the "Singletons" footer

    for _ in range(_MAX_CARDS):
        bg = FancyBboxPatch(
            (0.03, 0.0), 0.94, 0.12,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor="white", edgecolor="#cccccc", lw=1.2,
            transform=ax_cards.transAxes, alpha=0.0,
        )
        ax_cards.add_patch(bg)
        card_bg_patches.append(bg)
        card_texts.append({
            "title": ax_cards.text(0, 0, "", fontsize=10, fontweight="bold",
                                   color="#222", transform=ax_cards.transAxes,
                                   va="center", alpha=0.0),
            "line1": ax_cards.text(0, 0, "", fontsize=8.5, color="#333",
                                   transform=ax_cards.transAxes,
                                   va="center", alpha=0.0),
            "line2": ax_cards.text(0, 0, "", fontsize=8.5, color="#555",
                                   transform=ax_cards.transAxes,
                                   va="center", alpha=0.0),
            "line3": ax_cards.text(0, 0, "", fontsize=8.0, color="#777",
                                   transform=ax_cards.transAxes,
                                   va="center", alpha=0.0),
        })

    # ------------------------------------------------------------------ distance matrix
    ax_matrix.set_title(
        "Step D: Inter-AMR min predicted distance matrix over 5 s horizon  "
        f"(coupling threshold = {cluster_spatial_dist:.1f} m)",
        fontsize=10.5, pad=6,
    )
    ax_matrix.set_facecolor("#fafafa")
    # Initial placeholder heat-map (will be replaced each frame).
    _dummy = np.zeros((num_amrs, num_amrs))
    _amr_labels = [a.name for a in amrs]
    heat_img = ax_matrix.imshow(
        _dummy, vmin=0.0, vmax=cluster_spatial_dist * 2,
        cmap="RdYlGn", aspect="auto", origin="upper",
        interpolation="nearest",
    )
    plt.colorbar(heat_img, ax=ax_matrix, fraction=0.025, pad=0.01,
                 label="Min predicted distance [m]")
    ax_matrix.set_xticks(range(num_amrs))
    ax_matrix.set_yticks(range(num_amrs))
    ax_matrix.set_xticklabels(_amr_labels, fontsize=9)
    ax_matrix.set_yticklabels(_amr_labels, fontsize=9)
    ax_matrix.set_xlabel("AMR (column)")
    ax_matrix.set_ylabel("AMR (row)")

    # Cell text annotations for the matrix (distance values).
    _cell_texts = []
    for i in range(num_amrs):
        row_t = []
        for j in range(num_amrs):
            t = ax_matrix.text(j, i, "", ha="center", va="center",
                               fontsize=8.0, color="black", fontweight="bold")
            row_t.append(t)
        _cell_texts.append(row_t)

    # Threshold line overlay (drawn once).
    _threshold_line = ax_matrix.axhline(
        -0.5, color=_EDGE_COLOR, lw=0, alpha=0)   # invisible placeholder

    # ------------------------------------------------------------------ legend
    legend_handles = [
        plt.Line2D([0], [0], color="#bdbdbd", lw=6.0, alpha=0.65),
        Patch(facecolor=_rgba("#888888", 0.12),
              edgecolor="#888888", linewidth=1.4,
              linestyle="--"),
        Patch(facecolor=_rgba(CLUSTER_PALETTE[0], 0.18),
              edgecolor=CLUSTER_PALETTE[0], linewidth=2.5),
        plt.Line2D([0], [0], color=_EDGE_COLOR, lw=_EDGE_LW),
        Patch(facecolor=_rgba("#888888", 0.0),
              edgecolor=STATUS_COLOR["REPLAN"], linewidth=2.5),
        Patch(facecolor=_rgba("#888888", 0.0),
              edgecolor=STATUS_COLOR["SLOWDOWN"], linewidth=2.5),
        Patch(facecolor=_rgba("#888888", 0.0),
              edgecolor=STATUS_COLOR["WATCH"], linewidth=2.5),
        Patch(facecolor=_rgba("#888888", 0.0),
              edgecolor=STATUS_COLOR["CLEAR"], linewidth=2.5),
        Rectangle((0, 0), 1, 1, facecolor="#d32f2f", alpha=0.75),
        Rectangle((0, 0), 1, 1, facecolor="#388e3c", alpha=0.75),
    ]
    legend_labels = [
        "AMR reference rail",
        "Worker safety tube (Step B)",
        "Cluster local-replanning region (Step D hull)",
        f"Space-time coupling edge (pred-dist < {cluster_spatial_dist:.1f} m)",
        "AMR halo / ghost: REPLAN",
        "AMR halo / ghost: SLOWDOWN",
        "AMR halo / ghost: WATCH",
        "AMR halo / ghost: CLEAR",
        f"Matrix: coupled  (dist < {cluster_spatial_dist:.1f} m)",
        "Matrix: safe (green)",
    ]
    ax_leg.legend(legend_handles, legend_labels,
                  loc="center", ncol=5,
                  frameon=True, framealpha=0.95,
                  edgecolor="#cccccc", fontsize=8.5)

    # ================================================================== update
    def update(frame: int):
        # ----- Step A: worker prediction -----
        worker_data = []
        for w, predictor, art in zip(workers, predictors, worker_artists):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            out = predictor.rollout(obs)
            hard_lobes, soft_lobes, _ = safety.inflate_all(
                out["ellipses"], dt, out["belief"], mean_traj=out["mean"],
            )
            hull = safety_tube_polygon(hard_lobes)
            art["tube"].set_xy(hull)
            art["hist"].set_data(w["truth"][:frame+1, 0],
                                 w["truth"][:frame+1, 1])
            art["dot"].center = tuple(w["truth"][frame])
            art["tag"].set_position((float(w["truth"][frame, 0]),
                                     float(w["truth"][frame, 1]) + 0.42))
            worker_data.append(dict(
                name=w["name"], color=w["color"],
                inflated=hard_lobes, soft=soft_lobes,
                centers=out["ellipses"][:, :2],
            ))

        # ----- Step C: planner + conflict check -----
        planner.resolve(amrs, frame)

        results: dict[str, ConflictResult] = {}
        for amr in amrs:
            if not amr.is_active(frame):
                continue
            # Check for physical collision first.
            hit, hw = amr_human_collision(amr, frame, workers, human_collision_dist)
            if hit:
                amr.mark_collision(frame, hw)
                continue
            r = ConflictChecker.check(
                amr, dt, horizon_T, worker_data,
                t_replan=t_replan, v_amr_typical=v_amr_typical,
            )
            results[amr.name] = r

        # ----- Step D: cluster construction -----
        cluster_result = cluster_builder.build(
            amrs, results, frame, dt, horizon_T,
        )

        # ----- Advance AMRs -----
        for amr in amrs:
            if amr.is_active(frame):
                amr.step(dt)

        # ================================================================ render map

        # --- worker tubes already updated above ---

        # --- AMR bodies, halos, ghosts ---
        for ai, (amr, art) in enumerate(zip(amrs, amr_artists)):
            spawned = amr.is_spawned(frame)
            done = amr.is_done()
            collided = amr.collided

            if collided:
                cx, cy = amr.collision_pos
                head = amr.collision_heading
                art["body"].set_xy(_amr_body_polygon(cx, cy, head))
                art["body"].set_alpha(0.35)
                art["halo"].center = (cx, cy)
                art["halo"].set_edgecolor(STATUS_COLOR["COLLISION"])
                art["halo"].set_alpha(0.85)
                art["planned"].set_data([], [])
                for g in art["ghosts"]:
                    g.set_alpha(0.0)
                ltx, lty = _amr_tag_position(cx, cy, head)
                art["tag"].set_position((ltx, lty))
                art["tag"].set_alpha(0.65)
                continue

            if not spawned or done:
                art["body"].set_alpha(0.0)
                art["halo"].set_alpha(0.0)
                art["planned"].set_data([], [])
                for g in art["ghosts"]:
                    g.set_alpha(0.0)
                art["tag"].set_alpha(0.0)
                continue

            r = results.get(amr.name)
            status = r.status if r else "CLEAR"
            status_clr = STATUS_COLOR[status]
            t_replan_eff = r.t_replan_eff if r else t_replan
            alpha_amr = r.alpha_amr if r else 1.0

            cx, cy = amr.position_at(amr.progress)
            head = amr.heading_at(amr.progress)
            art["body"].set_xy(_amr_body_polygon(cx, cy, head))
            art["body"].set_facecolor(amr.color)
            art["body"].set_alpha(1.0)
            art["halo"].center = (cx, cy)
            art["halo"].set_edgecolor(status_clr)
            art["halo"].set_linewidth(2.8 if status == "REPLAN" else 1.8)
            art["halo"].set_alpha(0.90 if status == "REPLAN" else 0.55)

            if r is not None:
                art["planned"].set_data(
                    r.pred_positions[:, 0], r.pred_positions[:, 1])
                for k, t_idx in enumerate(ghost_idx):
                    gp = r.pred_positions[t_idx]
                    t_val = (t_idx + 1) * dt
                    g = art["ghosts"][k]
                    g.center = (float(gp[0]), float(gp[1]))
                    g.set_facecolor(_rgba(amr.color, 0.88 - 0.10 * k))
                    g.set_edgecolor(_ghost_border_color(
                        bool(r.hard_mask[t_idx]),
                        bool(r.soft_mask[t_idx]),
                        t_val, t_replan_eff, alpha_amr,
                    ))
                    g.set_radius(max(0.22 - 0.023 * k, 0.10))
                    g.set_alpha(0.90 - 0.07 * k)
            else:
                art["planned"].set_data([], [])
                for g in art["ghosts"]:
                    g.set_alpha(0.0)

            ltx, lty = _amr_tag_position(cx, cy, head)
            art["tag"].set_position((ltx, lty))
            art["tag"].set_alpha(1.0)

        # --- cluster hulls & coupling edges ---
        all_coupling_pairs = []
        name_to_cluster_color: dict[str, str] = {}
        for ci, cl in enumerate(cluster_result.clusters):
            for n in cl.member_names:
                name_to_cluster_color[n] = cl.color
            all_coupling_pairs.extend([(p, cl.color) for p in cl.coupling_pairs])

        # Draw hulls.
        for ci in range(max_clusters):
            hull_patch = cluster_hulls[ci]
            lbl = cluster_labels[ci]
            if ci < len(cluster_result.clusters):
                cl = cluster_result.clusters[ci]
                hull_patch.set_xy(cl.hull)
                hull_patch.set_edgecolor(cl.color)
                hull_patch.set_facecolor(_rgba(cl.color, 0.10))
                hull_patch.set_linewidth(2.4)
                hull_patch.set_linestyle("--")
                hull_patch.set_alpha(1.0)
                centroid = cl.hull.mean(axis=0)
                lbl.set_position(centroid)
                lbl.set_text(f"C{cl.cluster_id}")
                lbl.get_bbox_patch().set_facecolor(cl.color)
                lbl.set_alpha(1.0)
            else:
                hull_patch.set_alpha(0.0)
                lbl.set_alpha(0.0)

        # Draw coupling edges.
        edge_idx = 0
        active_names_set = {a.name for a in amrs if a.is_active(frame)}
        for (na, nb), clr in all_coupling_pairs:
            if edge_idx >= max_edges:
                break
            ai_obj = next((a for a in amrs if a.name == na), None)
            bi_obj = next((a for a in amrs if a.name == nb), None)
            if ai_obj is None or bi_obj is None:
                continue
            pa = ai_obj.position_at(ai_obj.progress)
            pb = bi_obj.position_at(bi_obj.progress)
            coupling_lines[edge_idx].set_data([pa[0], pb[0]], [pa[1], pb[1]])
            coupling_lines[edge_idx].set_color(clr)
            coupling_lines[edge_idx].set_alpha(_EDGE_ALPHA)
            edge_idx += 1

        for k in range(edge_idx, max_edges):
            coupling_lines[k].set_alpha(0.0)

        # ================================================================ cluster cards
        n_clusters = len(cluster_result.clusters)
        n_singletons = len(cluster_result.singletons)

        total_cards = n_clusters + (1 if n_singletons else 0)
        top_margin = 0.06
        bottom_margin = 0.04
        gap = 0.012
        usable = 1.0 - top_margin - bottom_margin - gap * max(total_cards - 1, 0)
        card_h = (usable / max(total_cards, 1))

        for ci in range(_MAX_CARDS):
            bg = card_bg_patches[ci]
            txts = card_texts[ci]
            if ci >= total_cards:
                bg.set_alpha(0.0)
                for t in txts.values():
                    t.set_alpha(0.0)
                continue

            top = 1.0 - top_margin - ci * (card_h + gap)
            bot = top - card_h
            cy_mid = (top + bot) / 2

            bg.set_bounds(0.03, bot + 0.003, 0.94, card_h - 0.006)
            bg.set_alpha(1.0)

            if ci < n_clusters:
                cl = cluster_result.clusters[ci]
                bg.set_edgecolor(cl.color)
                bg.set_linewidth(2.0)

                ttc_str = f"{cl.min_ttc:.1f}s" if cl.min_ttc < 900 else "∞"
                w_str = ", ".join(cl.trigger_workers) if cl.trigger_workers else "—"

                txts["title"].set_position((0.07, top - card_h * 0.18))
                txts["title"].set_text(
                    f"● C{cl.cluster_id}   [{cl.worst_status}]   TTC={ttc_str}")
                txts["title"].set_color(cl.color)
                txts["title"].set_alpha(1.0)

                txts["line1"].set_position((0.07, top - card_h * 0.43))
                txts["line1"].set_text(
                    "Members: " + "  ".join(cl.member_names))
                txts["line1"].set_alpha(1.0)

                txts["line2"].set_position((0.07, top - card_h * 0.65))
                txts["line2"].set_text(f"Worker: {w_str}")
                txts["line2"].set_alpha(1.0)

                bx, by, bw, bh = cl.bbox
                txts["line3"].set_position((0.07, top - card_h * 0.84))
                txts["line3"].set_text(
                    f"Replan box  {bw:.1f}×{bh:.1f} m  "
                    f"@ ({bx:.1f},{by:.1f})"
                )
                txts["line3"].set_alpha(1.0)

            else:
                # Singletons footer card.
                bg.set_edgecolor(SINGLETON_COLOR)
                bg.set_linewidth(1.2)
                s_str = ", ".join(cluster_result.singletons) or "—"
                txts["title"].set_position((0.07, top - card_h * 0.28))
                txts["title"].set_text("Singletons  (affected, no coupling)")
                txts["title"].set_color(SINGLETON_COLOR)
                txts["title"].set_alpha(1.0)
                txts["line1"].set_position((0.07, top - card_h * 0.65))
                txts["line1"].set_text(s_str)
                txts["line1"].set_alpha(1.0)
                txts["line2"].set_alpha(0.0)
                txts["line3"].set_alpha(0.0)

        # ================================================================ distance matrix
        active_names = cluster_result.active_names
        full_dist = np.full((num_amrs, num_amrs), np.nan)
        full_idx = {a.name: i for i, a in enumerate(amrs)}

        if len(active_names) > 0:
            for ii, ni in enumerate(active_names):
                for jj, nj in enumerate(active_names):
                    ri = full_idx.get(ni, -1)
                    rj = full_idx.get(nj, -1)
                    if ri >= 0 and rj >= 0:
                        full_dist[ri, rj] = cluster_result.dist_matrix[ii, jj]

        # Replace NaN (inactive AMRs) with a sentinel value above the cmap max.
        display_mat = np.where(np.isnan(full_dist),
                               cluster_spatial_dist * 2 + 0.1, full_dist)
        heat_img.set_data(display_mat)

        # Update cell text.
        for i in range(num_amrs):
            for j in range(num_amrs):
                val = full_dist[i, j]
                t = _cell_texts[i][j]
                if np.isnan(val):
                    t.set_text("—")
                    t.set_color("#aaaaaa")
                elif i == j:
                    t.set_text("0")
                    t.set_color("#666666")
                elif val < cluster_spatial_dist:
                    t.set_text(f"{val:.1f}")
                    t.set_color("#b71c1c")
                    t.set_fontweight("bold")
                else:
                    t.set_text(f"{val:.1f}")
                    t.set_color("#1b5e20")
                    t.set_fontweight("normal")

        # ================================================================ info text
        n_active = sum(1 for a in amrs if a.is_active(frame))
        n_replan = sum(1 for r in results.values() if r.status == "REPLAN")
        n_slow   = sum(1 for r in results.values() if r.status == "SLOWDOWN")
        info_text.set_text(
            f"frame {frame+1}/{num_frames}    "
            f"active {n_active}    "
            f"REPLAN {n_replan}  SLOWDOWN {n_slow}    "
            f"clusters {n_clusters}  singletons {n_singletons}\n"
            f"Step D: space-time coupling (dist < {cluster_spatial_dist:.1f} m)  "
            f"→  group affected AMRs  →  local replan region"
        )

        if not preview and (frame % 5 == 0 or frame == num_frames - 1):
            print(f"  rendering frame {frame+1}/{num_frames}")

        all_artists = [info_text, heat_img]
        for art in worker_artists:
            all_artists += [art["tube"], art["hist"], art["dot"], art["tag"]]
        for art in amr_artists:
            all_artists += [art["body"], art["halo"], art["planned"], art["tag"]]
            all_artists += art["ghosts"]
        all_artists += cluster_hulls + cluster_labels + coupling_lines
        all_artists += card_bg_patches
        for txts in card_texts:
            all_artists += list(txts.values())
        all_artists += [_cell_texts[i][j]
                        for i in range(num_amrs) for j in range(num_amrs)]
        return all_artists

    anim = FuncAnimation(fig, update, frames=num_frames,
                         interval=100, blit=False)

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
    except Exception as exc:
        print(f"  ffmpeg unavailable ({exc}); falling back to GIF")
        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=8))
        saved = gif_path
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
