"""Autonomous evaluation of Step E (V0).

Runs the full A->B->C->D->E(V0) pipeline headlessly over multiple seeds,
compares against the no-shield baseline, and auto-tunes the V0 shield
parameters by grid search (the deterministic-baseline stand-in for "training").

Usage::

    python -m Centralized_Local_Planner.eval_v0                # default 6 AMR / 2 workers
    python -m Centralized_Local_Planner.eval_v0 --frames 280 --seeds 5

Outputs a metrics table to stdout and writes ../outputs/step_e_v0_results.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .main import Pipeline
from .tools.replanning import V0Replanner, ReplanConfig


AMR_AMR_COLLISION_DIST = 0.60   # m, centre-to-centre physical overlap


def _outputs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs"


def run_episode(
    seed: int,
    num_frames: int,
    num_workers: int,
    num_amrs: int,
    replan_cfg: ReplanConfig | None,
) -> dict:
    """One headless episode. replan_cfg=None -> baseline (no V0 shield)."""
    rp = V0Replanner(replan_cfg) if replan_cfg is not None else None
    pipe = Pipeline(
        num_frames=num_frames, num_workers=num_workers,
        num_amrs=num_amrs, seed=seed, replanner=rp,
    )

    min_clear = float("inf")
    amr_amr_pairs: set[tuple[str, str]] = set()
    stop_count = 0
    active_count = 0

    for f in range(num_frames):
        pipe.step(f)
        active = [a for a in pipe.amrs if a.is_spawned(f) and not a.collided and not a.is_done()]
        for a in active:
            active_count += 1
            if a.actual_speed < 1e-6:
                stop_count += 1
        # worker clearance
        for a in active:
            pa = a.position_at(a.progress)
            for w in pipe.workers:
                if f < len(w["truth"]):
                    d = float(np.linalg.norm(pa - w["truth"][f]))
                    if d < min_clear:
                        min_clear = d
        # AMR-AMR physical overlap
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                pi = active[i].position_at(active[i].progress)
                pj = active[j].position_at(active[j].progress)
                if float(np.linalg.norm(pi - pj)) < AMR_AMR_COLLISION_DIST:
                    amr_amr_pairs.add(tuple(sorted((active[i].name, active[j].name))))

    worker_collisions = sum(1 for a in pipe.amrs if a.collided)
    completion = float(np.mean([a.is_done() for a in pipe.amrs]))
    mean_progress = float(np.mean([a.progress / max(a.total_length, 1e-9) for a in pipe.amrs]))
    stop_ratio = stop_count / max(active_count, 1)

    return dict(
        worker_collisions=int(worker_collisions),
        amr_amr_collisions=int(len(amr_amr_pairs)),
        completion=completion,
        mean_progress=mean_progress,
        min_clearance=float(min_clear),
        stop_ratio=float(stop_ratio),
    )


def aggregate(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def run_suite(seeds, num_frames, num_workers, num_amrs, replan_cfg) -> dict:
    rows = [run_episode(s, num_frames, num_workers, num_amrs, replan_cfg) for s in seeds]
    agg = aggregate(rows)
    agg["per_seed"] = rows
    return agg


def fmt(agg: dict) -> str:
    return (
        f"worker_coll={agg['worker_collisions']:.2f}  "
        f"amr_amr_coll={agg['amr_amr_collisions']:.2f}  "
        f"completion={agg['completion']*100:5.1f}%  "
        f"progress={agg['mean_progress']*100:5.1f}%  "
        f"min_clear={agg['min_clearance']:.2f}m  "
        f"stop_ratio={agg['stop_ratio']*100:4.1f}%"
    )


def main(argv: list[str] | None = None) -> None:
    pa = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--frames", type=int, default=280)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
    pa.add_argument("--tune-seeds", type=int, default=3, help="seeds used in the param sweep")
    args = pa.parse_args(argv)

    seeds = list(range(args.seeds))
    tune_seeds = list(range(args.tune_seeds))
    t0 = time.time()

    print(f"=== Step E (V0) evaluation :: {args.amrs} AMRs, {args.workers} workers, "
          f"{args.frames} frames ===\n")

    # ---- baseline (no shield) ----
    print("[1/3] Baseline (no Step-E shield) ...")
    baseline = run_suite(seeds, args.frames, args.workers, args.amrs, None)
    print("      " + fmt(baseline) + "\n")

    # ---- auto-tune V0 shield params (grid search) ----
    print("[2/3] Auto-tuning V0 shield (grid search over shield_steps x reserve_clearance) ...")
    grid_shield = [6, 8, 10, 12, 16, 20, 25]
    grid_clear = [1.0]
    sweep = []
    for ss, rc in itertools.product(grid_shield, grid_clear):
        cfg = ReplanConfig(shield_steps=ss, reserve_clearance=rc)
        agg = run_suite(tune_seeds, args.frames, args.workers, args.amrs, cfg)
        sweep.append((ss, rc, agg))
        print(f"      shield_steps={ss:2d} clearance={rc:.1f}  ->  " + fmt(agg))

    # Selection: fewest total collisions, then highest completion, then less stopping.
    def score(item):
        _, _, a = item
        return (a["worker_collisions"] + a["amr_amr_collisions"],
                -a["completion"], a["stop_ratio"])
    best_ss, best_rc, _ = min(sweep, key=score)
    best_cfg = ReplanConfig(shield_steps=best_ss, reserve_clearance=best_rc)
    print(f"\n      -> best config: shield_steps={best_ss}, reserve_clearance={best_rc}\n")

    # ---- final V0 evaluation with best config ----
    print("[3/3] V0 (tuned) full evaluation ...")
    v0 = run_suite(seeds, args.frames, args.workers, args.amrs, best_cfg)
    print("      " + fmt(v0) + "\n")

    # ---- summary ----
    print("=" * 78)
    print(f"{'metric':<22}{'baseline':>16}{'V0 (tuned)':>16}{'delta':>16}")
    print("-" * 78)
    for k, label in [
        ("worker_collisions", "worker collisions"),
        ("amr_amr_collisions", "AMR-AMR collisions"),
        ("completion", "completion %"),
        ("mean_progress", "path progress %"),
        ("min_clearance", "min clearance [m]"),
        ("stop_ratio", "stop ratio %"),
    ]:
        b, v = baseline[k], v0[k]
        scale = 100.0 if k in ("completion", "mean_progress", "stop_ratio") else 1.0
        print(f"{label:<22}{b*scale:>16.2f}{v*scale:>16.2f}{(v-b)*scale:>+16.2f}")
    print("=" * 78)
    print(f"total wall time: {time.time()-t0:.1f}s")

    out = _outputs_dir() / "step_e_v0_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        config=dict(frames=args.frames, workers=args.workers, amrs=args.amrs,
                    seeds=seeds, best_shield_steps=best_ss,
                    best_reserve_clearance=best_rc),
        baseline=baseline, v0=v0,
        sweep=[dict(shield_steps=ss, reserve_clearance=rc, **{k: a[k] for k in a if k != "per_seed"})
               for (ss, rc, a) in sweep],
    ), indent=2))
    print(f"saved results -> {out}")


if __name__ == "__main__":
    main()
