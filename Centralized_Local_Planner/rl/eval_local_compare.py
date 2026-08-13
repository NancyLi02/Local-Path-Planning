"""Compare spatial local replanning: RULE vs RL (learned attention policy),
both in the FULL simulator, over several seeds. Same 2D shield behind both.

    python -m Centralized_Local_Planner.rl.eval_local_compare --model logs/V1_local/episodic_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..tools.local_replanning import LocalReplanSim, LocalReplanConfig
from .local_cluster_env import LocalClusterEnv, LocalEnvConfig, OBS_DIM
from .local_policy import LocalAttentionPolicy
from .cluster_policy import act_clusters


def _min_clear_and_localframes(sim, f, acc):
    for a in sim.amrs:
        if a.is_spawned(f) and not a.collided and not a.is_done():
            p = a.current_xy()
            if a.local_mode:
                acc["localframes"] += 1
            for w in sim.workers:
                wp = w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]
                acc["min_clear"] = min(acc["min_clear"], float(np.linalg.norm(p - wp)))


def run_rule(seed, frames, workers, amrs):
    sim = LocalReplanSim(num_frames=frames, num_workers=workers, num_amrs=amrs,
                         seed=seed, cfg=LocalReplanConfig(exit_tol=0.6))
    acc = {"min_clear": float("inf"), "localframes": 0}
    for f in range(frames):
        sim.step(f); _min_clear_and_localframes(sim, f, acc)
    return _metrics(sim, acc, frames)


def run_rl(seed, frames, workers, amrs, policy, device):
    env = LocalClusterEnv(LocalEnvConfig(num_frames=frames, num_workers=workers, num_amrs=amrs))
    obs = env.reset(seed); sim = env.sim
    acc = {"min_clear": float("inf"), "localframes": 0}
    done = False
    while not done:
        act, _, _ = act_clusters(policy, obs, env.cluster_mask, env.cluster_valid,
                                 device, deterministic=True)
        obs, r, done, info = env.step(act)
        _min_clear_and_localframes(sim, env.frame - 1, acc)
    return _metrics(sim, acc, frames)


def _metrics(sim, acc, frames):
    return dict(
        completion=float(np.mean([a.is_done() for a in sim.amrs])),
        collisions=int(sum(a.collided for a in sim.amrs)),
        min_clear=float(acc["min_clear"] if np.isfinite(acc["min_clear"]) else 0.0),
        local_frames=acc["localframes"],
    )


def agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--model", type=str, default="logs/V1_local/episodic_best.pt")
    pa.add_argument("--seeds", type=int, default=5)
    pa.add_argument("--frames", type=int, default=420)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--hidden", type=int, default=64)
    args = pa.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seeds))
    policy = LocalAttentionPolicy(OBS_DIM, hidden=args.hidden).to(device)
    model = args.model if Path(args.model).is_absolute() else str(repo / args.model)
    policy.load_state_dict(torch.load(model, map_location=device)); policy.eval()

    print(f"device={device} seeds={seeds} model={args.model}\n")
    rule = agg([run_rule(s, args.frames, args.workers, args.amrs) for s in seeds])
    rl = agg([run_rl(s, args.frames, args.workers, args.amrs, policy, device) for s in seeds])

    print("=" * 74)
    print(f"{'metric':<22}{'rule-spatial':>16}{'RL-spatial':>16}{'delta':>16}")
    print("-" * 74)
    for k, label, sc in [("collisions", "worker collisions", 1.0),
                         ("completion", "completion %", 100.0),
                         ("min_clear", "min clearance [m]", 1.0),
                         ("local_frames", "AMR-frames replanning", 1.0)]:
        print(f"{label:<22}{rule[k]*sc:>16.2f}{rl[k]*sc:>16.2f}{(rl[k]-rule[k])*sc:>+16.2f}")
    print("=" * 74)
    out = repo / "outputs" / "step_e_local_results.json"
    out.write_text(json.dumps(dict(seeds=seeds, rule=rule, rl=rl, model=args.model), indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
