"""Demo renderer -- one animation, any of the four local replanning methods.

Animates the closed loop: worker prediction + safety tubes (Steps A/B), the
conflict-cluster busy areas (Step D), and for every AMR under local control the
commanded trajectory, the executed action `(goal_fwd, goal_lat, speed_scale)`
and whether the safety shield overrode the proposal.

    python -m Centralized_Local_Planner.local_path_replanning.render \
           --method optimization_based
    python -m Centralized_Local_Planner.local_path_replanning.render \
           --method learning_based --model logs/local_path_replanning/learning_based/best.pt

Rendering every method through this one file is what makes the demos
comparable: same camera, same panel, same colours, only the planner differs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle, Polygon

from ..viz.render_common import (
    _rgba, _draw_factory, _amr_body_polygon, _amr_tag_position,
)
from .common.config import PlannerConfig
from .registry import (
    DEFAULT_MODEL, METHODS, METHOD_TITLES, make_planner, resolve,
    runs_in_shared_runtime,
)
from .common.runtime import LocalReplanningRuntime
from .common.worker_cache import worker_frames

_REPO = Path(__file__).resolve().parents[2]
_TRAIL = 30
MAX_BUSY = 4
SHIELD_COLOR = "#ff5722"
CTRL_COLOR = "#00bcd4"


def simulate(method, cfg, frames, workers, amrs, seed, model, device):
    """Drive one method for `frames` frames and return drawable snapshots.

    `speed_adjusting` predates the shared runtime and drives its own
    pipeline, so it produces its snapshots through its adapter; every
    other method goes through LocalReplanningRuntime here.
    """
    method = resolve(method)
    if not runs_in_shared_runtime(method):
        from .speed_adjusting.adapter import run as run_speed_adjusting
        _, snaps = run_speed_adjusting(seed=seed, frames=frames, workers=workers,
                                       amrs=amrs, snapshots=True)
        return snaps, None

    wf = worker_frames(frames, workers, seed)
    planner = make_planner(method, cfg, model, device)
    rt = LocalReplanningRuntime(planner, cfg, num_frames=frames, num_workers=workers,
                      num_amrs=amrs, seed=seed, worker_frames=wf)
    trails = defaultdict(lambda: deque(maxlen=_TRAIL))
    prev_xy, snaps = {}, []
    for f in range(frames):
        out = rt.step(f)
        log = out["log"]
        amr_rows = []
        for a in rt.amrs:
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
            cmd = log.commands.get(a.name)
            st = rt.control.get(a.name)
            amr_rows.append(dict(
                name=a.name, color=a.color, xy=xy.copy(), heading=heading,
                controlled=st is not None, collided=a.collided, done=a.is_done(),
                spawned=spawned, trail=list(trails[a.name]),
                waypoints=(cmd.waypoints.copy() if cmd is not None else None),
                action=(np.asarray(cmd.action).copy() if cmd is not None and cmd.action is not None else None),
                mode=(cmd.mode if cmd is not None else ""),
                shield=(bool(cmd.shield_modified) if cmd is not None else False),
                unsafe=(not cmd.safe if cmd is not None else False),
                lat=(float(st.lat) if st is not None else 0.0),
            ))
        worker_rows = []
        for w, wd in zip(rt.workers, out["worker_data"]):
            worker_rows.append(dict(
                name=w["name"], color=w["color"],
                pos=(w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]),
                hard=wd["inflated"][0].copy(), tube=wd["tube"].copy()))
        snaps.append(dict(frame=f, amrs=amr_rows, workers=worker_rows,
                          locks=[lk["hull"].copy() for lk in out["locks"]],
                          plan_ms=log.plan_time_ms, n_ctrl=log.n_controlled))
    return snaps, rt




def build(snaps, rt, planner_kind, out_path, fps=12):
    del rt                                  # snapshots carry everything drawn
    fig, (ax, panel) = plt.subplots(
        1, 2, figsize=(17.5, 8.4), gridspec_kw={"width_ratios": [3.0, 1.0]})
    fig.patch.set_facecolor("#fbfbfb")
    _draw_factory(ax)
    ax.set_title(f"Local path replanning — "
             f"{METHOD_TITLES.get(planner_kind, planner_kind)}",
             fontsize=13, weight="bold")
    panel.axis("off")

    tubes = [ax.fill([], [], color="#9e9e9e", alpha=0.10, zorder=2)[0] for _ in range(4)]
    hards = [ax.fill([], [], color="#e53935", alpha=0.22, zorder=3)[0] for _ in range(4)]
    wdots = [ax.plot([], [], "o", ms=9, zorder=9)[0] for _ in range(4)]
    busy = [ax.fill([], [], color="#ffb300", alpha=0.18, zorder=5,
                    ec="#ef6c00", lw=2.2, ls="--")[0] for _ in range(MAX_BUSY)]

    n_amr = len(snaps[0]["amrs"])
    bodies, tags, trails_l, cmds, rings = [], [], [], [], []
    for row in snaps[0]["amrs"]:
        bodies.append(ax.fill([], [], color=row["color"], zorder=10, ec="#212121", lw=1.0)[0])
        tags.append(ax.text(0, 0, "", fontsize=9.5, zorder=11, ha="center",
                            color="#212121", weight="bold"))
        trails_l.append(ax.plot([], [], "-", color=row["color"], lw=1.2, alpha=0.35, zorder=4)[0])
        cmds.append(ax.plot([], [], "-o", color=CTRL_COLOR, lw=1.8, ms=2.6, alpha=0.9, zorder=8)[0])
        rings.append(Circle((0, 0), 0.62, fill=False, lw=0.0, zorder=12))
        ax.add_patch(rings[-1])
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#ef6c00", ls="--", lw=2.2, label="busy area (Step-D cluster)"),
        Line2D([], [], color="#e53935", lw=6, alpha=0.35, label="worker no-go lobe (Step B)"),
        Line2D([], [], color=CTRL_COLOR, lw=1.8, marker="o", ms=3,
               label="commanded trajectory"),
        Line2D([], [], color=SHIELD_COLOR, lw=2.2, ls="-", label="shield overrode proposal"),
    ], loc="lower left", fontsize=8, framealpha=0.9)

    hud = ax.text(0.01, 0.985, "", transform=ax.transAxes, va="top", fontsize=9,
                  family="monospace", bbox=dict(fc="white", ec="#bdbdbd", alpha=0.85))
    ptext = panel.text(0.0, 0.98, "", va="top", fontsize=9.5, family="monospace")

    def update(i):
        s = snaps[i]
        for k, art in enumerate(tubes):
            if k < len(s["workers"]):
                art.set_xy(s["workers"][k]["tube"])
            else:
                art.set_xy(np.zeros((1, 2)))
        for k, art in enumerate(hards):
            if k < len(s["workers"]):
                art.set_xy(s["workers"][k]["hard"])
            else:
                art.set_xy(np.zeros((1, 2)))
        for k, art in enumerate(wdots):
            if k < len(s["workers"]):
                w = s["workers"][k]
                art.set_data([w["pos"][0]], [w["pos"][1]]); art.set_color(w["color"])
            else:
                art.set_data([], [])
        for k, art in enumerate(busy):
            if k < len(s["locks"]):
                art.set_xy(s["locks"][k])
            else:
                art.set_xy(np.zeros((1, 2)))

        lines = [f"{'AMR':<7}{'mode':<7}{'fwd':>5}{'lat':>6}{'spd':>5}  shield"]
        for k, row in enumerate(s["amrs"]):
            if not row["spawned"] or row["done"]:
                bodies[k].set_xy(np.zeros((1, 2))); tags[k].set_text("")
                trails_l[k].set_data([], []); cmds[k].set_data([], [])
                rings[k].set_linewidth(0.0)
                continue
            x, y = float(row["xy"][0]), float(row["xy"][1])
            bodies[k].set_xy(_amr_body_polygon(x, y, row["heading"]))
            bodies[k].set_facecolor("#212121" if row["collided"] else row["color"])
            tx, ty = _amr_tag_position(x, y, row["heading"])
            tags[k].set_position((tx, ty)); tags[k].set_text(row["name"].replace("AMR-", ""))
            if row["trail"]:
                t = np.asarray(row["trail"]); trails_l[k].set_data(t[:, 0], t[:, 1])
            if row["waypoints"] is not None:
                wp = row["waypoints"]; cmds[k].set_data(wp[:, 0], wp[:, 1])
            else:
                cmds[k].set_data([], [])
            rings[k].center = (x, y)
            if row["controlled"]:
                rings[k].set_linewidth(2.2)
                rings[k].set_edgecolor(SHIELD_COLOR if row["shield"] else CTRL_COLOR)
            else:
                rings[k].set_linewidth(0.0)
            if row["controlled"] and row["action"] is not None:
                a = row["action"]
                flag = "OVERRIDE" if row["unsafe"] else ("backup" if row["shield"] else "")
                lines.append(f"{row['name']:<7}{row['mode']:<7}{a[0]:>5.1f}{a[1]:>+6.2f}"
                             f"{a[2]:>5.2f}  {flag}")
        hud.set_text(f"frame {s['frame']:3d}   under local control: {s['n_ctrl']}   "
                     f"plan {s['plan_ms']:5.1f} ms")
        ptext.set_text("\n".join(lines) if len(lines) > 1 else
                       "no cluster under local replanning")
        return []

    anim = FuncAnimation(fig, update, frames=len(snaps), interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Quality-targeted encode rather than a fixed bitrate: the panel text and
    # the thin trajectory lines are exactly what a low bitrate smears.
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-crf", "18", "-preset", "slow",
                                      "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    # dpi must keep the pixel size EVEN on both axes: libx264 with yuv420p
    # rejects an odd width, and matplotlib hands it figsize * dpi directly.
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    return out_path


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--method", choices=list(METHODS),
                    default="optimization_based")
    pa.add_argument("--model", default=None)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--frames", type=int, default=300)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--fps", type=int, default=12)
    pa.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a PlannerConfig field (repeatable)")
    pa.add_argument("--label", default=None,
                    help="override the demo file name")
    pa.add_argument("--out", default=None)
    args = pa.parse_args(argv)

    matplotlib.use("Agg")
    cfg = PlannerConfig()
    for item in args.set:
        key, _, value = item.partition("=")
        cur = getattr(cfg, key)
        cast = type(cur)
        setattr(cfg, key, value == "True" if cast is bool else cast(value))
    cfg.__post_init__()
    cfg.validate()
    method = resolve(args.method)
    model = args.model
    if model is None and method == "learning_based":
        model = DEFAULT_MODEL
    label = args.label or method
    snaps, rt = simulate(method, cfg, args.frames, args.workers, args.amrs,
                         args.seed, model, args.device)
    out = (Path(args.out) if args.out else
           _REPO / "outputs" / "5_local_path_replanning" / "demos" / f"demo_{label}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    build(snaps, rt, method, out, fps=args.fps)
    print(f"saved -> {out}")
    if rt is not None:
        m = rt.metrics()
        print("  completion %.0f%%  collisions %d  stop %.1f%%  "
              "shield %.1f%%  plan %.1f ms"
              % (m["completion"] * 100, m["worker_collisions"],
                 m["stop_ratio"] * 100, m["shield_rate"] * 100, m["plan_ms"]))


if __name__ == "__main__":
    main()
