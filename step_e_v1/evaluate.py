"""Evaluation harness for Step E: V0 vs V1 (and config ablations).

    python -m step_e_v1.evaluate --planner v0 --seeds 5
    python -m step_e_v1.evaluate --planner v1 --model logs/step_e_v1/best.pt --seeds 5
    python -m step_e_v1.evaluate --compare --model logs/step_e_v1/best.pt

``--set key=value`` overrides any V1Config field, which is how the ablation
studies are run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import V1Config
from .runtime import StepERuntime
from .v0_planner import V0SequentialReplanner
from .worker_cache import worker_frames

_REPO = Path(__file__).resolve().parents[1]

METRIC_ORDER = [
    ("worker_collisions", "worker collisions", 1.0, "%8.2f"),
    ("completion", "completion %", 100.0, "%8.1f"),
    ("progress", "path progress %", 100.0, "%8.1f"),
    ("min_clearance", "min clearance [m]", 1.0, "%8.2f"),
    ("makespan", "makespan [frames]", 1.0, "%8.0f"),
    ("stop_ratio", "stop ratio %", 100.0, "%8.1f"),
    ("shield_rate", "shield override %", 100.0, "%8.1f"),
    ("unsafe_rate", "no-safe-candidate %", 100.0, "%8.1f"),
    ("route_deviation", "route deviation [m]", 1.0, "%8.3f"),
    ("control_frames", "AMR-frames replanned", 1.0, "%8.0f"),
    ("candidates_per_amr", "rollouts per AMR", 1.0, "%8.2f"),
    ("plan_ms", "plan time [ms]", 1.0, "%8.1f"),
]


def make_planner(kind: str, cfg: V1Config, model: str | None = None,
                 device: str = "cpu"):
    if kind == "v0":
        return V0SequentialReplanner(cfg)
    if kind == "stopgo":
        from .stopgo_planner import StopAndGoReplanner
        return StopAndGoReplanner(cfg)
    if kind == "v1":
        from .v1_planner import V1AttentionReplanner
        from .attention_policy import MultiAMRAttentionPolicy
        if model is None:
            return V1AttentionReplanner(MultiAMRAttentionPolicy(cfg), cfg, device=device)
        path = model if Path(model).is_absolute() else str(_REPO / model)
        return V1AttentionReplanner.from_checkpoint(path, cfg, device=device)
    raise ValueError(kind)


def run_seed(kind: str, cfg: V1Config, seed: int, frames: int, num_workers: int,
             num_amrs: int, model: str | None = None, device: str = "cpu") -> dict:
    wf = worker_frames(frames, num_workers, seed)
    planner = make_planner(kind, cfg, model, device)
    rt = StepERuntime(planner, cfg, num_frames=frames, num_workers=num_workers,
                      num_amrs=num_amrs, seed=seed, worker_frames=wf)
    return rt.run(frames)


def run_config(kind: str, cfg: V1Config, seeds, frames: int = 420,
               num_workers: int = 2, num_amrs: int = 6, model: str | None = None,
               device: str = "cpu") -> dict:
    rows = [run_seed(kind, cfg, s, frames, num_workers, num_amrs, model, device)
            for s in seeds]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["per_seed"] = rows
    return agg


def print_table(columns: dict) -> None:
    names = list(columns)
    head = f"{'metric':<24}" + "".join(f"{n:>16}" for n in names)
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for key, label, scale, fmt in METRIC_ORDER:
        row = f"{label:<24}"
        for n in names:
            row += f"{columns[n].get(key, float('nan')) * scale:>16.3f}"
        print(row)
    print("=" * len(head))


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--planner", choices=["stopgo", "v0", "v1"], default="v0")
    pa.add_argument("--planners", type=str, default=None,
                    help="comma-separated list, e.g. stopgo,v0,v1")
    pa.add_argument("--compare", action="store_true", help="run V0 and V1 side by side")
    pa.add_argument("--model", type=str, default=None)
    pa.add_argument("--seeds", type=int, default=5)
    pa.add_argument("--frames", type=int, default=420)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a V1Config field (repeatable)")
    pa.add_argument("--out", type=str, default=None)
    args = pa.parse_args(argv)

    cfg = V1Config()
    for item in args.set:
        key, _, value = item.partition("=")
        cur = getattr(cfg, key)
        cast = type(cur)
        setattr(cfg, key, value == "True" if cast is bool else cast(value))
    cfg.__post_init__()
    cfg.validate()

    seeds = list(range(args.seeds))
    if args.planners:
        kinds = [k.strip() for k in args.planners.split(",") if k.strip()]
    elif args.compare:
        kinds = ["v0", "v1"]
    else:
        kinds = [args.planner]
    columns = {}
    _LABEL = {"stopgo": "STOP-GO", "v0": "V0", "v1": "V1"}
    for kind in kinds:
        columns[_LABEL.get(kind, kind.upper())] = run_config(kind, cfg, seeds, args.frames, args.workers,
                                           args.amrs, args.model, args.device)
    print(f"\nseeds={seeds} frames={args.frames} amrs={args.amrs} workers={args.workers}")
    for item in args.set:
        print(f"  override {item}")
    print_table(columns)

    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else _REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({k: {kk: vv for kk, vv in v.items()}
                                   for k, v in columns.items()}, indent=2, default=float))
        print(f"saved -> {out}")
    return columns


if __name__ == "__main__":
    main()
