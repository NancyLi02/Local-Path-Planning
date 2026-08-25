"""The original rail Step-E V0, run as-is, reported and drawn like the rest.

This module does NOT re-implement anything. It calls the existing planner --
`Centralized_Local_Planner.tools.replanning.V0Replanner` driving
`Centralized_Local_Planner.main.Pipeline` -- and only

  * collects the same metrics vocabulary the other rungs report, and
  * emits the frame snapshots `step_e_v1.render.build` draws,

so the speed-only rung can sit in the same table and the same set of videos
without being reinterpreted.

    python -m step_e_v1.legacy_rail --seeds 5 --frames 560
    python -m step_e_v1.legacy_rail --render --seed 0 --frames 420

What the original planner does: order the active AMRs by TTC, try the speed
factors 1, 2/3, 1/3, 0 fastest-first, commit the first whose 5 s rail rollout
stays out of every worker's Step-B hard lobe and out of the higher-priority
AMRs' reservations, reserve it, and STOP when none is admissible. Every active
AMR is shielded every frame; there is no cluster hand-over and no busy area.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from Centralized_Local_Planner.main import Pipeline
from Centralized_Local_Planner.tools.geometry import safety_tube_polygon
from Centralized_Local_Planner.tools.replanning import (
    ACTION_GO, ACTION_STOP, ReplanConfig, V0Replanner,
)

_REPO = Path(__file__).resolve().parents[1]
_TRAIL = 30

# The tuned configuration this planner was evaluated with (see
# outputs/5_step_e_rail_v0_speed/results.json).
TUNED = ReplanConfig(shield_steps=25, reserve_clearance=1.0)


class _TimedV0(V0Replanner):
    """The original planner, with the planning call timed so the frame time
    of the (slow) particle predictor does not get charged to it."""

    last_ms: float = 0.0

    def plan(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = super().plan(*args, **kwargs)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return out


def run(seed: int = 0, frames: int = 560, workers: int = 2, amrs: int = 6,
        cfg: ReplanConfig | None = None, snapshots: bool = False):
    """Run one episode. Returns (metrics, snapshots|None)."""
    replanner = _TimedV0(cfg or TUNED)
    pipe = Pipeline(num_frames=frames, num_workers=workers, num_amrs=amrs,
                    seed=seed, replanner=replanner)

    trails = defaultdict(lambda: deque(maxlen=_TRAIL))
    prev_xy: dict[str, np.ndarray] = {}
    snaps: list[dict] = []
    done_frame: dict[str, int] = {}
    min_clear = float("inf")
    stop_count = active_count = 0
    slow_count = 0
    plan_ms: list[float] = []

    for f in range(frames):
        out = pipe.step(f)
        plan_ms.append(replanner.last_ms)

        actions = dict(replanner.last_actions)
        speeds = dict(replanner.last_speeds)

        rows = []
        for a in pipe.amrs:
            spawned = a.is_spawned(f)
            active = spawned and not a.collided and not a.is_done()
            xy = a.position_at(a.progress)
            if active:
                active_count += 1
                if a.actual_speed < 1e-6:
                    stop_count += 1
                elif actions.get(a.name) != ACTION_GO:
                    slow_count += 1
                for w in pipe.workers:
                    wp = w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]
                    min_clear = min(min_clear, float(np.linalg.norm(xy - wp)))
            if a.name not in done_frame and a.is_done():
                done_frame[a.name] = f
            if not snapshots:
                continue

            if active:
                trails[a.name].append((float(xy[0]), float(xy[1])))
            pv = prev_xy.get(a.name)
            if pv is not None and float(np.linalg.norm(xy - pv)) > 1e-4:
                heading = float(np.arctan2(xy[1] - pv[1], xy[0] - pv[0]))
            else:
                heading = a.heading_at(a.progress)
            prev_xy[a.name] = xy

            v = float(speeds.get(a.name, a.actual_speed))
            factor = v / max(a.commanded_speed, 1e-9)
            label = actions.get(a.name, "")
            way = None
            if active:
                way = np.array([a.position_at(a.progress + v * (t + 1) * pipe.dt)
                                for t in range(0, TUNED.shield_steps, 3)])
            rows.append(dict(
                name=a.name, color=a.color, xy=xy.copy(), heading=heading,
                controlled=active,                  # every AMR is shielded here
                collided=a.collided, done=a.is_done(), spawned=spawned,
                trail=list(trails[a.name]), waypoints=way,
                # `fwd` shows the distance this speed is checked over -- the
                # shield's 5 s rail rollout -- so the panel stays meaningful
                # even though this planner has no goal_fwd / goal_lat.
                action=np.array([v * TUNED.shield_steps * pipe.dt, 0.0, factor]),
                mode=(ACTION_STOP if label == ACTION_STOP else "TRACK"),
                shield=(label not in ("", ACTION_GO)),
                unsafe=False, lat=0.0,
            ))

        if snapshots:
            wrows = []
            for w, wd in zip(pipe.workers, out["worker_data"]):
                wrows.append(dict(
                    name=w["name"], color=w["color"],
                    pos=(w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]),
                    hard=wd["inflated"][0].copy(),
                    tube=wd.get("tube", safety_tube_polygon(wd["inflated"])).copy()))
            n_ctrl = sum(1 for r in rows if r["controlled"])
            snaps.append(dict(frame=f, amrs=rows, workers=wrows, locks=[],
                              plan_ms=plan_ms[-1], n_ctrl=n_ctrl))

    finished = all(a.is_done() for a in pipe.amrs)
    metrics = dict(
        worker_collisions=float(sum(a.collided for a in pipe.amrs)),
        completion=float(np.mean([a.is_done() for a in pipe.amrs])),
        progress=float(np.mean([a.progress / a.total_length for a in pipe.amrs])),
        min_clearance=float(min_clear if np.isfinite(min_clear) else 0.0),
        makespan=float(max(done_frame.values()) if finished and done_frame else frames),
        makespan_censored=float(0.0 if finished else 1.0),
        stop_ratio=float(stop_count / max(active_count, 1)),
        shield_rate=float((stop_count + slow_count) / max(active_count, 1)),
        control_frames=float(active_count),
        candidates_per_amr=float(len(TUNED.speed_factors)),
        plan_ms=float(np.mean(plan_ms)),
    )
    return metrics, (snaps if snapshots else None)


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--seeds", type=int, default=5)
    pa.add_argument("--seed", type=int, default=0, help="single seed, for --render")
    pa.add_argument("--frames", type=int, default=560)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--render", action="store_true")
    pa.add_argument("--fps", type=int, default=12)
    pa.add_argument("--out", default=None)
    args = pa.parse_args(argv)

    if args.render:
        import matplotlib
        matplotlib.use("Agg")
        from .render import build
        metrics, snaps = run(args.seed, args.frames, args.workers, args.amrs,
                             snapshots=True)
        out = Path(args.out) if args.out else (
            _REPO / "outputs" / "8_step_e_v1_module" / "demos" / "demo_rail_v0.mp4")
        build(snaps, None, "rail_v0", out, fps=args.fps)
        print(f"saved -> {out}")
        print("  completion %.0f%%  collisions %.0f  stop %.1f%%  plan %.1f ms"
              % (metrics["completion"] * 100, metrics["worker_collisions"],
                 metrics["stop_ratio"] * 100, metrics["plan_ms"]))
        return metrics

    rows = [run(s, args.frames, args.workers, args.amrs)[0] for s in range(args.seeds)]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"\nseeds={list(range(args.seeds))} frames={args.frames} "
          f"amrs={args.amrs} workers={args.workers}   (original rail V0)")
    print("=" * 44)
    for k, label, scale in [("worker_collisions", "worker collisions", 1.0),
                            ("completion", "completion %", 100.0),
                            ("progress", "path progress %", 100.0),
                            ("min_clearance", "min clearance [m]", 1.0),
                            ("makespan", "makespan [frames]", 1.0),
                            ("stop_ratio", "stop ratio %", 100.0),
                            ("shield_rate", "shield slowed/stopped %", 100.0),
                            ("candidates_per_amr", "rollouts per AMR", 1.0),
                            ("plan_ms", "frame time [ms]", 1.0)]:
        print(f"{label:<26}{agg[k] * scale:>16.3f}")
    print("=" * 44)
    out = Path(args.out) if args.out else (
        _REPO / "outputs" / "8_step_e_v1_module" / "results" / "rail_v0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(agg, per_seed=rows), indent=2, default=float))
    print(f"saved -> {out}")
    return agg


if __name__ == "__main__":
    main()
