"""Step E demo animation -- replanning actions + per-AMR Gantt status.

`_simulate` drives V0 (Pipeline + V0Replanner) into per-frame snapshots;
`animate_snapshots` turns snapshots into the figure (map with action-coloured
AMRs + worker no-go lobes, bottom Gantt, HUD). The animation routine is shared:
the V1 demo (rl/render_v1.py) produces snapshots in the same format and reuses it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon, Rectangle

from ..main import Pipeline
from ..tools.replanning import V0Replanner, ReplanConfig
from ..tools.geometry import safety_tube_polygon
from .render_common import (
    _rgba, _draw_factory, _amr_body_polygon, _amr_tag_position, STATUS_COLOR,
)

ACTION_COLOR = {"GO": "#2ca02c", "SLOW": "#f9a825", "STOP": "#d62728"}
COLLIDED_COLOR = "#212121"
DONE_COLOR = "#bdbdbd"


def _gantt_colors(result, horizon_T, dt):
    """Per-look-ahead-step status colours: hard conflict (red) / soft (watch) / clear."""
    colors = []
    for t in range(horizon_T):
        if result.hard_mask[t]:
            c = STATUS_COLOR["REPLAN"]
        elif result.soft_mask[t]:
            c = STATUS_COLOR["WATCH"]
        else:
            c = STATUS_COLOR["CLEAR"]
        colors.append(c)
    return colors


# ---------------------------------------------------------------------------
# Shared snapshot -> animation
# ---------------------------------------------------------------------------

def animate_snapshots(amr_meta, worker_meta, snaps, dt, horizon_T, num_frames,
                      output_path, preview, title, shield_seconds):
    """Render per-frame snapshots. amr_meta:[{name,color,waypoints}],
    worker_meta:[{name,color}], snaps: list of frame dicts (see _simulate)."""
    total_amrs = len(amr_meta)
    names = [m["name"] for m in amr_meta]
    horizon_seconds = horizon_T * dt

    fig = plt.figure(figsize=(15.5, 10.6), dpi=120)
    gs = fig.add_gridspec(3, 1, height_ratios=[12.0, 4.6, 1.0], hspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    ax_gantt = fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[2, 0]); ax_leg.axis("off")

    _draw_factory(ax)
    ax.set_title(title, fontsize=13, pad=10)

    for m in amr_meta:
        wp = m["waypoints"]
        ax.plot(wp[:, 0], wp[:, 1], "-", color="#bdbdbd", lw=5.5, alpha=0.32,
                solid_capstyle="round", zorder=1)
        ax.plot(wp[:, 0], wp[:, 1], "--", color=m["color"], lw=1.0, alpha=0.45, zorder=1)

    placeholder = np.array([[0.0, 0.0], [0.0, 1e-3], [1e-3, 0.0]])
    worker_art = []
    for w in worker_meta:
        tube = Polygon(placeholder, closed=True, facecolor=_rgba(w["color"], 0.07),
                       edgecolor=_rgba(w["color"], 0.40), lw=1.0, zorder=2)
        hard = Polygon(placeholder, closed=True, facecolor="none",
                       edgecolor="#cc0033", lw=2.0, linestyle="-", zorder=4)
        ax.add_patch(tube); ax.add_patch(hard)
        dot = Circle((0, 0), 0.22, facecolor=w["color"], edgecolor="white", lw=1.0, zorder=6)
        ax.add_patch(dot)
        tag = ax.text(0, 0, w["name"], fontsize=8.5, color="white", fontweight="bold",
                      ha="center", va="bottom",
                      bbox=dict(facecolor=w["color"], edgecolor="white", lw=0.6,
                                pad=2, boxstyle="round,pad=0.18"), zorder=7)
        worker_art.append(dict(tube=tube, hard=hard, dot=dot, tag=tag))

    amr_art = []
    for m in amr_meta:
        body = Polygon(placeholder, closed=True, facecolor=m["color"],
                       edgecolor="white", lw=1.4, alpha=0.0, zorder=6)
        halo = Circle((0, 0), 0.66, facecolor="none", edgecolor=ACTION_COLOR["GO"],
                      lw=2.4, alpha=0.0, zorder=5)
        ax.add_patch(body); ax.add_patch(halo)
        tag = ax.text(0, 0, "", fontsize=7.5, color=m["color"], fontweight="bold",
                      ha="center", va="center", alpha=0.0, zorder=8)
        amr_art.append(dict(body=body, halo=halo, tag=tag))

    hud = ax.text(0.4, 11.6, "", fontsize=9.5, color="#222", va="top", ha="left",
                  bbox=dict(facecolor="white", alpha=0.88, edgecolor="#cccccc"), zorder=9)

    # ----- bottom Gantt chart -----
    ax_gantt.set_xlim(0, horizon_seconds)
    ax_gantt.set_ylim(-0.5, total_amrs - 0.5)
    ax_gantt.invert_yaxis()
    ax_gantt.set_yticks(range(total_amrs))
    ax_gantt.set_yticklabels(names, fontsize=9)
    ax_gantt.set_xlabel("Lookahead horizon  t  [s]")
    ax_gantt.set_title(
        f"Per-AMR space-time conflict status  —  brakes on any hard cell "
        f"within the {shield_seconds:.1f}s shield",
        fontsize=10.5, pad=6)
    ax_gantt.set_facecolor("#fafafa")
    ax_gantt.grid(True, alpha=0.30, axis="x")
    for spine in ("top", "right"):
        ax_gantt.spines[spine].set_visible(False)
    if shield_seconds < horizon_seconds - 1e-6:
        ax_gantt.axvline(shield_seconds, color="#1565c0", lw=1.8, linestyle="-", alpha=0.7)
        ax_gantt.text(shield_seconds - 0.05, total_amrs - 0.45,
                      f"shield = {shield_seconds:.1f}s",
                      fontsize=8.0, color="#1565c0", ha="right", va="bottom", fontweight="bold")

    gantt_cells, gantt_overlays, gantt_action = [], [], []
    for ai in range(total_amrs):
        row = []
        for t in range(horizon_T):
            rect = Rectangle((t * dt, ai - 0.36), dt, 0.72,
                             facecolor=STATUS_COLOR["PENDING"], edgecolor="white", lw=0.4)
            ax_gantt.add_patch(rect); row.append(rect)
        gantt_cells.append(row)
        overlay = Rectangle((0.0, ai - 0.36), horizon_seconds, 0.72, facecolor="none",
                            edgecolor=COLLIDED_COLOR, hatch="//", linewidth=0.0,
                            alpha=0.0, zorder=4)
        ax_gantt.add_patch(overlay); gantt_overlays.append(overlay)
        badge = ax_gantt.text(horizon_seconds * 1.012, ai, "", fontsize=8,
                              fontweight="bold", color="white", ha="left", va="center",
                              bbox=dict(facecolor=ACTION_COLOR["GO"], edgecolor="none",
                                        boxstyle="round,pad=0.22"), alpha=0.0, zorder=7)
        gantt_action.append(badge)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor=ACTION_COLOR["GO"], markersize=12, markeredgewidth=2.4),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor=ACTION_COLOR["SLOW"], markersize=12, markeredgewidth=2.4),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor=ACTION_COLOR["STOP"], markersize=12, markeredgewidth=2.4),
        plt.Line2D([0], [0], color="#cc0033", lw=2.0),
        Rectangle((0, 0), 1, 1, facecolor=STATUS_COLOR["REPLAN"]),
        Rectangle((0, 0), 1, 1, facecolor=STATUS_COLOR["WATCH"]),
        Rectangle((0, 0), 1, 1, facecolor=STATUS_COLOR["CLEAR"]),
    ]
    labels = ["GO (full speed)", "SLOW", "STOP (yield)", "Worker hard no-go lobe",
              "Gantt: hard conflict (brake)", "Gantt: soft (watch)", "Gantt: clear"]
    ax_leg.legend(handles, labels, loc="center", ncol=7, frameon=True,
                  framealpha=0.95, edgecolor="#cccccc", fontsize=8.5,
                  columnspacing=1.0, handletextpad=0.5)

    def update(frame: int):
        s = snaps[frame]
        for wa, ws in zip(worker_art, s["workers"]):
            wa["tube"].set_xy(ws["tube"]); wa["hard"].set_xy(ws["hard0"])
            wa["dot"].center = tuple(ws["pos"])
            wa["tag"].set_position((float(ws["pos"][0]), float(ws["pos"][1]) + 0.42))
        for ai, (aa, as_) in enumerate(zip(amr_art, s["amrs"])):
            if not as_["spawned"]:
                aa["body"].set_alpha(0.0); aa["halo"].set_alpha(0.0); aa["tag"].set_alpha(0.0)
            else:
                cx, cy = float(as_["pos"][0]), float(as_["pos"][1])
                aa["body"].set_xy(_amr_body_polygon(cx, cy, as_["heading"]))
                if as_["collided"]:
                    aa["body"].set_facecolor(COLLIDED_COLOR); aa["body"].set_alpha(0.9)
                    aa["halo"].set_edgecolor(COLLIDED_COLOR); aa["halo"].set_alpha(0.9)
                elif as_["done"]:
                    aa["body"].set_facecolor(as_["color"]); aa["body"].set_alpha(0.25)
                    aa["halo"].set_alpha(0.0)
                else:
                    aa["body"].set_facecolor(as_["color"]); aa["body"].set_alpha(1.0)
                    aa["halo"].set_edgecolor(ACTION_COLOR.get(as_["action"], ACTION_COLOR["GO"]))
                    aa["halo"].set_alpha(0.95)
                aa["halo"].center = (cx, cy)
                tx, ty = _amr_tag_position(cx, cy, as_["heading"])
                aa["tag"].set_position((tx, ty)); aa["tag"].set_text(as_["name"]); aa["tag"].set_alpha(0.9)
            for t, rect in enumerate(gantt_cells[ai]):
                rect.set_facecolor(as_["gantt"][t])
            gantt_overlays[ai].set_alpha(0.55 if as_["collided"] else 0.0)
            badge = gantt_action[ai]
            if as_["collided"]:
                badge.set_text("CRASH"); badge.get_bbox_patch().set_facecolor(COLLIDED_COLOR)
                badge.set_alpha(1.0)
            elif as_["spawned"] and not as_["done"]:
                act = as_["action"]; badge.set_text(act)
                badge.get_bbox_patch().set_facecolor(ACTION_COLOR.get(act, ACTION_COLOR["GO"]))
                badge.set_alpha(1.0)
            else:
                badge.set_alpha(0.0)
        hud.set_text(
            f"frame {s['frame']+1}/{num_frames}\n"
            f"AMR-worker collisions: {s['collisions']}\n"
            f"completed: {s['done']}/{total_amrs}\n"
            f"min clearance: {s['min_clear']:.2f} m")
        if not preview and (frame % 20 == 0 or frame == num_frames - 1):
            print(f"  rendering frame {frame+1}/{num_frames}")
        arts = [hud]
        for wa in worker_art:
            arts += [wa["tube"], wa["hard"], wa["dot"], wa["tag"]]
        for aa in amr_art:
            arts += [aa["body"], aa["halo"], aa["tag"]]
        for row in gantt_cells:
            arts += row
        arts += gantt_overlays + gantt_action
        return arts

    anim = FuncAnimation(fig, update, frames=len(snaps), interval=100, blit=False)
    if preview:
        print("Preview mode: showing live window. Close it to exit.")
        plt.show(); return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(output_path, writer=FFMpegWriter(fps=12, bitrate=2400))
        saved = output_path
    except Exception as exc:                                # pragma: no cover
        print(f"  ffmpeg unavailable ({exc}); falling back to GIF")
        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=10)); saved = gif_path
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# V0 simulation -> snapshots
# ---------------------------------------------------------------------------

def _simulate(num_frames, num_workers, num_amrs, seed, cfg):
    pipe = Pipeline(num_frames=num_frames, num_workers=num_workers,
                    num_amrs=num_amrs, seed=seed, replanner=V0Replanner(cfg))
    snaps = []
    min_clear = float("inf")
    last_gantt = {a.name: [STATUS_COLOR["PENDING"]] * pipe.horizon_T for a in pipe.amrs}
    collided_meta: dict[str, tuple] = {}
    for f in range(num_frames):
        out = pipe.step(f)
        actions = dict(pipe.replanner.last_actions)
        results = out["results"]
        amrs = []
        for a in pipe.amrs:
            pos = a.position_at(a.progress)
            spawned = a.is_spawned(f)
            if not a.collided and not a.is_done() and spawned:
                for w in pipe.workers:
                    if f < len(w["truth"]):
                        min_clear = min(min_clear, float(np.linalg.norm(pos - w["truth"][f])))
            if a.collided:
                collided_meta.setdefault(a.name, (a.collision_frame, a.collision_worker))
                gantt = last_gantt[a.name]
            elif a.name in results:
                gantt = _gantt_colors(results[a.name], pipe.horizon_T, pipe.dt)
                last_gantt[a.name] = gantt
            elif not spawned:
                gantt = [STATUS_COLOR["PENDING"]] * pipe.horizon_T
            elif a.is_done():
                gantt = [STATUS_COLOR["DONE"]] * pipe.horizon_T
            else:
                gantt = [STATUS_COLOR["CLEAR"]] * pipe.horizon_T
            amrs.append(dict(name=a.name, color=a.color, pos=pos,
                             heading=a.heading_at(a.progress),
                             action=actions.get(a.name, "GO"),
                             collided=a.collided, done=a.is_done(), spawned=spawned,
                             gantt=list(gantt), coll_meta=collided_meta.get(a.name)))
        workers = []
        for w, wd in zip(pipe.workers, out["worker_data"]):
            workers.append(dict(name=w["name"], color=w["color"],
                                pos=w["truth"][f] if f < len(w["truth"]) else w["truth"][-1],
                                hard0=np.asarray(wd["inflated"][0]),
                                tube=safety_tube_polygon(wd["inflated"])))
        snaps.append(dict(amrs=amrs, workers=workers, frame=f,
                          collisions=sum(1 for a in pipe.amrs if a.collided),
                          done=sum(1 for a in pipe.amrs if a.is_done()),
                          min_clear=(min_clear if np.isfinite(min_clear) else 0.0)))
    return pipe, snaps


def build_replanning_animation(output_path, num_frames=280, num_workers=2, num_amrs=6,
                               preview=False, seed=7, shield_steps=8, reserve_clearance=1.0):
    cfg = ReplanConfig(shield_steps=shield_steps, reserve_clearance=reserve_clearance)
    pipe, snaps = _simulate(num_frames, num_workers, num_amrs, seed, cfg)
    amr_meta = [dict(name=a.name, color=a.color, waypoints=a.waypoints) for a in pipe.amrs]
    worker_meta = [dict(name=w["name"], color=w["color"]) for w in pipe.workers]
    shield_seconds = min(shield_steps, pipe.horizon_T) * pipe.dt
    return animate_snapshots(
        amr_meta, worker_meta, snaps, pipe.dt, pipe.horizon_T, num_frames,
        output_path, preview,
        "Step E (V0): TTC-priority replanning + space-time reservation shield",
        shield_seconds)
