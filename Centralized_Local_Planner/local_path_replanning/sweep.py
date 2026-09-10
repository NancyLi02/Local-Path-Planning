"""Config sweep over the shield / planner design switches.

    python -m Centralized_Local_Planner.local_path_replanning.sweep --ablation --seeds 5 --jobs 5

Runs one method (by default ``optimization_based``, the deterministic method
that defines the shield's behaviour) over a grid of configuration overrides and
prints one row per configuration, so each design decision is settled by
measurement rather than by intuition.

``--ablation`` disables exactly one decision per row, which is the table that
belongs in the write-up; without it the product grid is an exploratory sweep.
"""
from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common.config import PlannerConfig
from .evaluate import run_config
from .registry import METHODS

_REPO = Path(__file__).resolve().parents[2]

# Product grid (exploratory).
GRID = dict(
    candidate_grid=[False, True],
    hard_horizon_sec=[1.0, 1.5],
    min_amr_distance=[0.9, 1.1],
)

# One-factor-at-a-time ablations: each row disables or reverts exactly one
# design decision relative to the default configuration, so the table reads as
# "what does this decision buy?".
ABLATIONS = [
    ("default", {}),
    ("shorter hard window (1.0 s)", dict(hard_horizon_sec=1.0)),
    ("no detour candidates", dict(use_detour_candidates=False)),
    ("no emergency reverse", dict(allow_emergency_reverse=False)),
    ("no command-consistency cost", dict(w_switch=0.0)),

    ("no lateral DOF (speed only)", dict(max_lateral_offset=0.01)),
    ("plain STOP fallback (no least-unsafe)", dict(least_unsafe_fallback=False)),
    ("with command-consistency cost", dict(w_switch=1.5)),
    ("tighter keep-out (0.85 m)", dict(worker_circle_clearance=0.85)),
    ("dense candidate grid", dict(candidate_grid=True)),
    ("backups off (proposal + STOP)", dict(use_backup_candidates=False)),
]


def _run(item):
    overrides, seeds, frames, amrs, workers, method, model = item
    cfg = PlannerConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__(); cfg.validate()
    res = run_config(method, cfg, seeds, frames, workers, amrs, model)
    res.pop("per_seed", None)
    return overrides, res


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--seeds", type=int, default=3)
    pa.add_argument("--frames", type=int, default=420)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--jobs", type=int, default=4)
    pa.add_argument("--method", choices=list(METHODS),
                    default="optimization_based")
    pa.add_argument("--model", default=None)
    pa.add_argument("--out", default="outputs/5_local_path_replanning/results/sweep.json")
    pa.add_argument("--ablation", action="store_true",
                    help="one-factor-at-a-time ablation instead of the product grid")
    args = pa.parse_args(argv)

    seeds = list(range(args.seeds))
    if args.ablation:
        labels = [name for name, _ in ABLATIONS]
        combos = [dict(o) for _, o in ABLATIONS]
    else:
        keys = list(GRID)
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]
        labels = [" ".join(f"{k.split('_')[0]}={v}" for k, v in c.items()) for c in combos]
    items = [(c, seeds, args.frames, args.amrs, args.workers, args.method, args.model)
             for c in combos]

    with Pool(processes=args.jobs) as pool:
        results = pool.map(_run, items)

    labelled = list(zip(labels, [r for _, r in results]))
    if not args.ablation:
        labelled.sort(key=lambda r: (r[1]["worker_collisions"], -r[1]["completion"],
                                     r[1]["stop_ratio"]))
    head = (f"{'config':<52}{'coll':>6}{'compl%':>8}{'prog%':>8}{'stop%':>7}"
            f"{'unsafe%':>9}{'dev':>7}{'minclr':>8}{'ms':>7}")
    print(head); print("-" * len(head))
    for label, r in labelled:
        print(f"{label:<52}{r['worker_collisions']:>6.2f}{r['completion']*100:>8.1f}"
              f"{r['progress']*100:>8.1f}{r['stop_ratio']*100:>7.1f}"
              f"{r['unsafe_rate']*100:>9.1f}{r['route_deviation']:>7.3f}"
              f"{r['min_clearance']:>8.2f}{r['plan_ms']:>7.1f}")

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([dict(label=l, config=o, **{k: v for k, v in r.items()})
                               for (l, r), (o, _) in zip(labelled, results)],
                              indent=2, default=float))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
