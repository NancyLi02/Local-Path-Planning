"""Evaluation harness: run any of the four local replanning methods, compare them.

    # one method
    python -m Centralized_Local_Planner.local_path_replanning.evaluate \
           --method optimization_based --seeds 5

    # the whole ladder, one table
    python -m Centralized_Local_Planner.local_path_replanning.evaluate \
           --methods stop_and_go,speed_adjusting,optimization_based,learning_based \
           --seeds 5 --frames 560 --out outputs/5_local_path_replanning/results/comparison.json

Every method is measured with the same metric vocabulary over the same seeds,
so the columns are directly comparable. ``--set key=value`` overrides any
``PlannerConfig`` field, which is how the ablation studies are run.

``speed_adjusting`` is measured inside its own pipeline (it is the original
rail planner and predates the shared runtime), so a few metrics are not defined
the same way for it; those are reported as NaN rather than silently
reinterpreted. See ``speed_adjusting/adapter.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common.config import PlannerConfig
from .common.runtime import LocalReplanningRuntime
from .common.worker_cache import worker_frames
from .registry import (
    DEFAULT_MODEL, METHODS, METHOD_LABELS, make_planner, resolve,
    runs_in_shared_runtime,
)

_REPO = Path(__file__).resolve().parents[2]

METRIC_ORDER = [
    ("worker_collisions", "worker collisions", 1.0),
    ("completion", "completion %", 100.0),
    ("progress", "path progress %", 100.0),
    ("min_clearance", "min clearance [m]", 1.0),
    ("makespan", "makespan [frames]", 1.0),
    ("stop_ratio", "stop ratio %", 100.0),
    ("shield_rate", "shield override %", 100.0),
    ("unsafe_rate", "no-safe-candidate %", 100.0),
    ("route_deviation", "route deviation [m]", 1.0),
    ("control_frames", "AMR-frames replanned", 1.0),
    ("candidates_per_amr", "rollouts per AMR", 1.0),
    ("plan_ms", "plan time [ms]", 1.0),
]


def run_seed(method: str, cfg: PlannerConfig, seed: int, frames: int,
             num_workers: int, num_amrs: int, model: str | None = None,
             device: str = "cpu") -> dict:
    """One episode of one method. Returns the metric dict."""
    method = resolve(method)

    if not runs_in_shared_runtime(method):
        from .speed_adjusting.adapter import run as run_speed_adjusting
        metrics, _ = run_speed_adjusting(seed=seed, frames=frames,
                                         workers=num_workers, amrs=num_amrs)
        return metrics

    wf = worker_frames(frames, num_workers, seed)
    planner = make_planner(method, cfg, model, device)
    rt = LocalReplanningRuntime(planner, cfg, num_frames=frames,
                                num_workers=num_workers, num_amrs=num_amrs,
                                seed=seed, worker_frames=wf)
    return rt.run(frames)


def run_config(method: str, cfg: PlannerConfig, seeds, frames: int = 420,
               num_workers: int = 2, num_amrs: int = 6, model: str | None = None,
               device: str = "cpu") -> dict:
    rows = [run_seed(method, cfg, s, frames, num_workers, num_amrs, model, device)
            for s in seeds]
    keys = sorted({k for r in rows for k in r})
    agg = {k: float(np.mean([r[k] for r in rows if k in r])) if any(k in r for r in rows)
           else float("nan") for k in keys}
    agg["per_seed"] = rows
    return agg


def print_table(columns: dict) -> None:
    names = list(columns)
    head = f"{'metric':<24}" + "".join(f"{n:>16}" for n in names)
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for key, label, scale in METRIC_ORDER:
        row = f"{label:<24}"
        for n in names:
            v = columns[n].get(key, float("nan"))
            row += f"{'n/a':>16}" if not np.isfinite(v) else f"{v * scale:>16.3f}"
        print(row)
    print("=" * len(head))


def main(argv=None):
    pa = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--method", choices=list(METHODS), default="optimization_based")
    pa.add_argument("--methods", type=str, default=None,
                    help="comma-separated list; the whole ladder is "
                         "stop_and_go,speed_adjusting,optimization_based,learning_based")
    pa.add_argument("--model", type=str, default=None,
                    help=f"learning_based checkpoint (default {DEFAULT_MODEL})")
    pa.add_argument("--seeds", type=int, default=5)
    pa.add_argument("--frames", type=int, default=420)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a PlannerConfig field (repeatable)")
    pa.add_argument("--out", type=str, default=None)
    args = pa.parse_args(argv)

    cfg = PlannerConfig()
    for item in args.set:
        key, _, value = item.partition("=")
        cast = type(getattr(cfg, key))
        setattr(cfg, key, value == "True" if cast is bool else cast(value))
    cfg.__post_init__()
    cfg.validate()

    seeds = list(range(args.seeds))
    methods = ([resolve(m.strip()) for m in args.methods.split(",") if m.strip()]
               if args.methods else [resolve(args.method)])
    model = args.model
    if model is None and "learning_based" in methods:
        model = DEFAULT_MODEL

    columns = {}
    for method in methods:
        columns[METHOD_LABELS[method]] = run_config(
            method, cfg, seeds, args.frames, args.workers, args.amrs, model,
            args.device)

    print(f"\nseeds={seeds} frames={args.frames} amrs={args.amrs} "
          f"workers={args.workers}")
    for item in args.set:
        print(f"  override {item}")
    print_table(columns)

    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else _REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(columns, indent=2, default=float))
        print(f"saved -> {out}")
    return columns


if __name__ == "__main__":
    main()
