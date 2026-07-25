"""Step E (spatial) demo -- cluster-level local PATH replanning.

Shows AMRs leaving their rails inside a locked BUSY AREA, routing around workers
and each other in 2D, then rejoining the reference path; other AMRs wait at the
busy-area boundary. Pre-simulates LocalReplanSim into snapshots, then animates.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon

from ..tools.local_replanning import LocalReplanSim, LocalReplanConfig
from ..tools.geometry import safety_tube_polygon
from .render_common import _rgba, _draw_factory, _amr_body_polygon, _amr_tag_position

RAIL_COLOR = {"rail": None}        # AMR uses its own colour on rail
LOCAL_EDGE = "#ff5722"             # highlight ring while local-replanning
COLLIDED_COLOR = "#212121"
MAX_BUSY = 4                       # patch pool size
_TRAIL = 30


def _simulate(num_frames, num_workers, num_amrs, seed):
    sim = LocalReplanSim(num_frames=num_frames, num_workers=num_workers,
                         num_amrs=num_amrs, seed=seed)
    trails = defaultdict(lambda: deque(maxlen=_TRAIL))
    prev_xy = {}
    snaps = []
    min_clear = float("inf")
    for f in range(num_frames):
        out = sim.step(f)
        amrs = []
        for a in sim.amrs:
            xy = a.current_xy()
            spawned = a.is_spawned(f)
            if spawned and not a.collided:
                trails[a.name].append((float(xy[0]), float(xy[1])))
            pv = prev_xy.get(a.name)
            if pv is not None and float(np.linalg.norm(xy - pv)) > 1e-4:
                heading = float(np.arctan2(xy[1] - pv[1], xy[0] - pv[0]))
            else:
                heading = a.heading_at(a.progress)
            prev_xy[a.name] = xy
            if spawned and not a.collided and not a.is_done():
                for w in sim.workers:
                    wp = w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]
                    min_clear = min(min_clear, float(np.linalg.norm(xy - wp)))
            amrs.append(dict(name=a.name, color=a.color, xy=xy, heading=heading,
                             local=a.local_mode, collided=a.collided,
                             done=a.is_done(), spawned=spawned,
                             trail=list(trails[a.name])))
        workers = []
        for w, wd in zip(sim.workers, out["worker_data"]):
            workers.append(dict(name=w["name"], color=w["color"],
                                pos=w["truth"][f] if f < len(w["truth"]) else w["truth"][-1],
                                hard0=np.asarray(wd["inflated"][0]),
                                tube=safety_tube_polygon(wd["inflated"])))
        busy = [dict(hull=lk.hull, color=lk.color) for lk in out["locks"]]
        snaps.append(dict(frame=f, amrs=amrs, workers=workers, busy=busy,
                          n_local=sum(1 for a in sim.amrs if a.local_mode),
                          done=sum(1 for a in sim.amrs if a.is_done()),
                          collisions=sum(1 for a in sim.amrs if a.collided),
                          min_clear=(min_clear if np.isfinite(min_clear) else 0.0)))
    return sim, snaps


def build_local_animation(output_path, num_frames=420, num_workers=2, num_amrs=6,
                          preview=False, seed=7):
    sim, snaps = _simulate(num_frames, num_workers, num_amrs, seed)
    return animate_local(
        sim, snaps, num_frames, output_path, preview,
        "Step E (spatial): cluster local PATH replanning in a locked busy area")


def animate_local(sim, snaps, num_frames, output_path, preview, title):
    """Render pre-built local-replanning snapshots (shared by rule + RL demos)."""
    fig = plt.figure(figsize=(15.5, 9.2), dpi=120)
    gs = fig.add_gridspec(2, 1, height_ratios=[16, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[1, 0]); ax_leg.axis("off")
    _draw_factory(ax)
    ax.set_title(title, fontsize=13, pad=10)

    for a in sim.amrs:
        ax.plot(a.waypoints[:, 0], a.waypoints[:, 1], "--",
                color=a.color, lw=1.0, alpha=0.45, zorder=1)

    placeholder = np.array([[0.0, 0.0], [0.0, 1e-3], [1e-3, 0.0]])
    busy_patches = [Polygon(placeholder, closed=True, facecolor="none",
                            edgecolor="none", alpha=0.0, zorder=1) for _ in range(MAX_BUSY)]
    for p in busy_patches:
        ax.add_patch(p)

    worker_art = []
    for w in sim.workers:
        tube = Polygon(placeholder, closed=True, facecolor=_rgba(w["color"], 0.07),
                       edgecolor=_rgba(w["color"], 0.40), lw=1.0, zorder=2)
        hard = Polygon(placeholder, closed=True, facecolor="none",
                       edgecolor="#cc0033", lw=2.0, zorder=4)
        ax.add_patch(tube); ax.add_patch(hard)
        dot = Circle((0, 0), 0.22, facecolor=w["color"], edgecolor="white", lw=1.0, zorder=7)
        ax.add_patch(dot)
        tag = ax.text(0, 0, w["name"], fontsize=8.5, color="white", fontweight="bold",
                      ha="center", va="bottom",
                      bbox=dict(facecolor=w["color"], edgecolor="white", lw=0.6,
                                pad=2, boxstyle="round,pad=0.18"), zorder=8)
        worker_art.append(dict(tube=tube, hard=hard, dot=dot, tag=tag))

    amr_art = []
    for a in sim.amrs:
        trail, = ax.plot([], [], "-", color=a.color, lw=2.0, alpha=0.7, zorder=3)
        body = Polygon(placeholder, closed=True, facecolor=a.color,
                       edgecolor="white", lw=1.4, alpha=0.0, zorder=6)
        halo = Circle((0, 0), 0.62, facecolor="none", edgecolor=LOCAL_EDGE,
                      lw=2.6, alpha=0.0, zorder=5)
        ax.add_patch(body); ax.add_patch(halo)
        tag = ax.text(0, 0, "", fontsize=7.5, color=a.color, fontweight="bold",
                      ha="center", va="center", alpha=0.0, zorder=9)
        amr_art.append(dict(trail=trail, body=body, halo=halo, tag=tag))

    hud = ax.text(0.4, 11.6, "", fontsize=9.5, color="#222", va="top", ha="left",
                  bbox=dict(facecolor="white", alpha=0.88, edgecolor="#cccccc"), zorder=10)

    handles = [
        Polygon(placeholder, closed=True, facecolor=_rgba("#e63946", 0.15),
                edgecolor="#e63946", lw=2.0),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor=LOCAL_EDGE, markersize=12, markeredgewidth=2.6),
        plt.Line2D([0], [0], color="#888", lw=2.0),
        plt.Line2D([0], [0], color="#cc0033", lw=2.0),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLLIDED_COLOR,
                   markersize=11, linestyle="None"),
    ]
    labels = ["Busy area (locked cluster region)", "AMR in local-replan mode",
              "AMR local path (trail)", "Worker hard no-go lobe", "Collision"]
    ax_leg.legend(handles, labels, loc="center", ncol=5, frameon=True,
                  framealpha=0.95, edgecolor="#cccccc", fontsize=9)

    def update(frame):
        s = snaps[frame]
        for i, p in enumerate(busy_patches):
            if i < len(s["busy"]):
                b = s["busy"][i]
                p.set_xy(b["hull"]); p.set_facecolor(_rgba(b["color"], 0.15))
                p.set_edgecolor(b["color"]); p.set_linewidth(2.0); p.set_alpha(1.0)
            else:
                p.set_alpha(0.0)
        for wa, ws in zip(worker_art, s["workers"]):
            wa["tube"].set_xy(ws["tube"]); wa["hard"].set_xy(ws["hard0"])
            wa["dot"].center = tuple(ws["pos"])
            wa["tag"].set_position((float(ws["pos"][0]), float(ws["pos"][1]) + 0.42))
        for aa, as_ in zip(amr_art, s["amrs"]):
            if not as_["spawned"]:
                aa["body"].set_alpha(0.0); aa["halo"].set_alpha(0.0)
                aa["tag"].set_alpha(0.0); aa["trail"].set_data([], [])
                continue
            cx, cy = float(as_["xy"][0]), float(as_["xy"][1])
            aa["body"].set_xy(_amr_body_polygon(cx, cy, as_["heading"]))
            if as_["collided"]:
                aa["body"].set_facecolor(COLLIDED_COLOR); aa["body"].set_alpha(0.9)
                aa["halo"].set_edgecolor(COLLIDED_COLOR); aa["halo"].set_alpha(0.9)
            elif as_["done"]:
                aa["body"].set_facecolor(as_["color"]); aa["body"].set_alpha(0.25)
                aa["halo"].set_alpha(0.0)
            else:
                aa["body"].set_facecolor(as_["color"]); aa["body"].set_alpha(1.0)
                aa["halo"].set_alpha(0.95 if as_["local"] else 0.0)
            aa["halo"].center = (cx, cy)
            if as_["local"] and as_["trail"]:
                tr = np.array(as_["trail"])
                aa["trail"].set_data(tr[:, 0], tr[:, 1])
            else:
                aa["trail"].set_data([], [])
            tx, ty = _amr_tag_position(cx, cy, as_["heading"])
            aa["tag"].set_position((tx, ty)); aa["tag"].set_text(as_["name"]); aa["tag"].set_alpha(0.9)
        hud.set_text(
            f"frame {s['frame']+1}/{num_frames}\n"
            f"in local-replan: {s['n_local']}    busy areas: {len(s['busy'])}\n"
            f"completed: {s['done']}/{len(sim.amrs)}    collisions: {s['collisions']}\n"
            f"min clearance: {s['min_clear']:.2f} m")
        if not preview and (frame % 20 == 0 or frame == num_frames - 1):
            print(f"  rendering frame {frame+1}/{num_frames}")
        arts = [hud] + busy_patches
        for wa in worker_art:
            arts += [wa["tube"], wa["hard"], wa["dot"], wa["tag"]]
        for aa in amr_art:
            arts += [aa["trail"], aa["body"], aa["halo"], aa["tag"]]
        return arts

    anim = FuncAnimation(fig, update, frames=len(snaps), interval=100, blit=False)
    if preview:
        plt.show(); return None
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(output_path, writer=FFMpegWriter(fps=12, bitrate=2400)); saved = output_path
    except Exception as exc:                                # pragma: no cover
        gif = output_path.with_suffix(".gif"); anim.save(gif, writer=PillowWriter(fps=10)); saved = gif
    plt.close(fig)
    return saved


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--frames", type=int, default=420)
    pa.add_argument("--seed", type=int, default=7)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--output", type=str, default="outputs/step_e_local_replanning_demo.mp4")
    a = pa.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = Path(a.output)
    if not out.is_absolute():
        out = repo / out
    saved = build_local_animation(out, a.frames, a.workers, a.amrs, seed=a.seed)
    print(f"VIDEO SAVED {saved}")
