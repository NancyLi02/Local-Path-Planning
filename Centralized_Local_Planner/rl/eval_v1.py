"""Evaluate V1 (learned attention policy) vs V0 (greedy factor=1) in FleetEnv.

Both run behind the identical V0 safety shield, so the comparison isolates the
*coordination* the policy adds. Metrics mirror eval_v0 for comparability.

    python -m Centralized_Local_Planner.rl.eval_v1 --model logs/V1/v1_best.pt --seeds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .fleet_env import FleetEnv, FleetEnvConfig, OBS_DIM
from .policy import AttentionPolicy

_AMR_AMR_DIST = 0.60


def run_episode(env, seed, action_fn):
    obs = env.reset(seed)
    min_clear = float("inf")
    stop_steps = 0
    active_steps = 0
    done = False
    while not done:
        mask = env.active_mask
        act = action_fn(obs, mask)
        obs, r, done, info = env.step(act)
        f = min(env.frame - 1, env.cfg.num_frames - 1)
        active = [a for a in env.amrs if a.is_spawned(env.frame - 1) and not a.collided and not a.is_done()]
        active_steps += len(active)
        for a in active:
            p = a.position_at(a.progress)
            if a.actual_speed < 1e-6:
                stop_steps += 1
            for w in env.workers:
                wp = w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]
                min_clear = min(min_clear, float(np.linalg.norm(p - wp)))
    return dict(
        worker_collisions=int(info["collided"]),
        completion=float(info["completion"]),
        progress=float(info["progress"]),
        min_clearance=float(min_clear if np.isfinite(min_clear) else 0.0),
        stop_ratio=stop_steps / max(active_steps, 1),
    )


def agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--model", type=str, default="logs/V1/v1_best.pt")
    pa.add_argument("--seeds", type=int, default=5)
    pa.add_argument("--frames", type=int, default=360)
    pa.add_argument("--hidden", type=int, default=64)
    args = pa.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seeds))
    env = FleetEnv(FleetEnvConfig(num_frames=args.frames))

    policy = AttentionPolicy(OBS_DIM, hidden=args.hidden).to(device)
    policy.load_state_dict(torch.load(repo / args.model, map_location=device))
    policy.eval()

    def v0_fn(obs, mask):
        return np.ones(env.N, dtype=np.float32)

    def v1_fn(obs, mask):
        ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        mt = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
        a, _, _ = policy.act(ot, mt, deterministic=True)
        return a[0].cpu().numpy()

    print(f"device={device}  seeds={seeds}  model={args.model}\n")
    v0 = agg([run_episode(env, s, v0_fn) for s in seeds])
    v1 = agg([run_episode(env, s, v1_fn) for s in seeds])

    print("=" * 70)
    print(f"{'metric':<22}{'V0 (greedy)':>14}{'V1 (learned)':>14}{'delta':>14}")
    print("-" * 70)
    rows = [("worker_collisions", "worker collisions", 1.0),
            ("completion", "completion %", 100.0),
            ("progress", "path progress %", 100.0),
            ("min_clearance", "min clearance [m]", 1.0),
            ("stop_ratio", "stop ratio %", 100.0)]
    for k, label, sc in rows:
        print(f"{label:<22}{v0[k]*sc:>14.2f}{v1[k]*sc:>14.2f}{(v1[k]-v0[k])*sc:>+14.2f}")
    print("=" * 70)

    out = repo / "outputs" / "step_e_v1_results.json"
    out.write_text(json.dumps(dict(seeds=seeds, v0=v0, v1=v1, model=args.model), indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
