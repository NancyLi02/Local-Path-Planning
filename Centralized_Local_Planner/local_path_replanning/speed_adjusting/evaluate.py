"""Tuning + baseline evaluation for the speed-adjusting planner.

Two things this answers that the cross-method comparison does not:

1. **Is the shield worth anything at all?** It runs the same fleet with no
   local replanning whatsoever (the AMRs just drive the QR-planner speed) and
   compares.
2. **What shield horizon should the planner use?** ``shield_steps`` is the one
   parameter that matters, and it is chosen by grid search rather than by
   assertion -- the deterministic stand-in for "training".

    python -m Centralized_Local_Planner.local_path_replanning.speed_adjusting.evaluate
    python -m Centralized_Local_Planner.local_path_replanning.speed_adjusting.evaluate \
           --frames 280 --seeds 5

The episode loop itself lives in ``adapter.run`` and is shared with the
cross-method harness, so the numbers here and in the comparison table come from
exactly the same code path.

Writes outputs/5_local_path_replanning/results/speed_adjusting_tuning.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .adapter import run
from .planner import SpeedAdjustingConfig

_REPO = Path(__file__).resolve().parents[3]

_ROWS = [
    ("worker_collisions", "worker collisions", 1.0),
    ("amr_amr_collisions", "AMR-AMR collisions", 1.0),
    ("completion", "completion %", 100.0),
    ("progress", "path progress %", 100.0),
    ("min_clearance", "min clearance [m]", 1.0),
    ("stop_ratio", "stop ratio %", 100.0),
]


def run_suite(seeds, frames, workers, amrs,
              cfg: SpeedAdjustingConfig | None, shielded: bool = True) -> dict:
    """Aggregate `adapter.run` over seeds. cfg=None + shielded=False = baseline."""
    rows = [run(seed=s, frames=frames, workers=workers, amrs=amrs,
                cfg=cfg, shielded=shielded)[0] for s in seeds]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["per_seed"] = rows
    return agg


def fmt(agg: dict) -> str:
    return (f"worker_coll={agg['worker_collisions']:.2f}  "
            f"amr_amr_coll={agg['amr_amr_collisions']:.2f}  "
            f"completion={agg['completion'] * 100:5.1f}%  "
            f"progress={agg['progress'] * 100:5.1f}%  "
            f"min_clear={agg['min_clearance']:.2f}m  "
            f"stop_ratio={agg['stop_ratio'] * 100:4.1f}%")


def main(argv: list[str] | None = None) -> None:
    pa = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--frames", type=int, default=280)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
    pa.add_argument("--tune-seeds", type=int, default=3,
                    help="seeds used in the parameter sweep")
    pa.add_argument("--out", type=str, default=None)
    args = pa.parse_args(argv)

    seeds = list(range(args.seeds))
    tune_seeds = list(range(args.tune_seeds))
    t0 = time.time()

    print(f"=== speed adjusting :: {args.amrs} AMRs, {args.workers} workers, "
          f"{args.frames} frames ===\n")

    print("[1/3] Baseline (no local replanning at all) ...")
    baseline = run_suite(seeds, args.frames, args.workers, args.amrs,
                         None, shielded=False)
    print("      " + fmt(baseline) + "\n")

    print("[2/3] Grid search over shield_steps x reserve_clearance ...")
    sweep = []
    for ss, rc in itertools.product([6, 8, 10, 12, 16, 20, 25], [1.0]):
        cfg = SpeedAdjustingConfig(shield_steps=ss, reserve_clearance=rc)
        agg = run_suite(tune_seeds, args.frames, args.workers, args.amrs, cfg)
        sweep.append((ss, rc, agg))
        print(f"      shield_steps={ss:2d} clearance={rc:.1f}  ->  " + fmt(agg))

    # Fewest collisions first, then highest completion, then least stopping.
    def score(item):
        _, _, a = item
        return (a["worker_collisions"] + a["amr_amr_collisions"],
                -a["completion"], a["stop_ratio"])

    best_ss, best_rc, _ = min(sweep, key=score)
    print(f"\n      -> best: shield_steps={best_ss}, reserve_clearance={best_rc}\n")

    print("[3/3] Tuned full evaluation ...")
    tuned = run_suite(seeds, args.frames, args.workers, args.amrs,
                      SpeedAdjustingConfig(shield_steps=best_ss,
                                           reserve_clearance=best_rc))
    print("      " + fmt(tuned) + "\n")

    print("=" * 78)
    print(f"{'metric':<22}{'no replanning':>16}{'tuned':>16}{'delta':>16}")
    print("-" * 78)
    for key, label, scale in _ROWS:
        b, v = baseline[key], tuned[key]
        print(f"{label:<22}{b * scale:>16.2f}{v * scale:>16.2f}"
              f"{(v - b) * scale:>+16.2f}")
    print("=" * 78)
    print(f"total wall time: {time.time() - t0:.1f}s")

    out = (Path(args.out) if args.out else
           _REPO / "outputs" / "5_local_path_replanning" / "results"
           / "speed_adjusting_tuning.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        config=dict(frames=args.frames, workers=args.workers, amrs=args.amrs,
                    seeds=seeds, best_shield_steps=best_ss,
                    best_reserve_clearance=best_rc),
        baseline=baseline, tuned=tuned,
        sweep=[dict(shield_steps=ss, reserve_clearance=rc,
                    **{k: v for k, v in a.items() if k != "per_seed"})
               for (ss, rc, a) in sweep],
    ), indent=2, default=float))
    print(f"saved results -> {out}")


if __name__ == "__main__":
    main()
