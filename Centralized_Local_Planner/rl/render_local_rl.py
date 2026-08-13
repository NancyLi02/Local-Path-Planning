"""Render the RL (Phase 2) spatial local-replanning demo: the trained attention
policy drives the in-cluster AMRs (residual local goals behind the 2D shield).

    python -m Centralized_Local_Planner.rl.render_local_rl --model logs/V1_local/local_best.pt
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from .local_cluster_env import LocalClusterEnv, LocalEnvConfig, OBS_DIM
from .local_policy import LocalAttentionPolicy
from .cluster_policy import act_clusters
from ..tools.geometry import safety_tube_polygon
from ..viz.render_local import animate_local, _TRAIL


def _simulate_rl(model, num_frames, num_workers, num_amrs, seed, hidden, device):
    env = LocalClusterEnv(LocalEnvConfig(num_frames=num_frames, num_workers=num_workers,
                                         num_amrs=num_amrs))
    policy = LocalAttentionPolicy(OBS_DIM, hidden=hidden).to(device)
    policy.load_state_dict(torch.load(model, map_location=device)); policy.eval()

    obs = env.reset(seed); sim = env.sim
    trails = defaultdict(lambda: deque(maxlen=_TRAIL)); prev_xy = {}
    snaps = []; min_clear = float("inf"); done = False
    for f in range(num_frames):
        if done:
            break
        act, _, _ = act_clusters(policy, obs, env.cluster_mask, env.cluster_valid,
                                 device, deterministic=True)
        obs, r, done, info = env.step(act)

        amrs = []
        for a in sim.amrs:
            xy = a.current_xy(); spawned = a.is_spawned(f)
            if spawned and not a.collided:
                trails[a.name].append((float(xy[0]), float(xy[1])))
            pv = prev_xy.get(a.name)
            heading = (float(np.arctan2(xy[1] - pv[1], xy[0] - pv[0]))
                       if pv is not None and float(np.linalg.norm(xy - pv)) > 1e-4
                       else a.heading_at(a.progress))
            prev_xy[a.name] = xy
            if spawned and not a.collided and not a.is_done():
                for w in sim.workers:
                    wp = w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]
                    min_clear = min(min_clear, float(np.linalg.norm(xy - wp)))
            amrs.append(dict(name=a.name, color=a.color, xy=xy, heading=heading,
                             local=a.local_mode, collided=a.collided,
                             done=a.is_done(), spawned=spawned, trail=list(trails[a.name])))
        workers = []
        for w, wd in zip(sim.workers, sim._worker_data(f)):
            workers.append(dict(name=w["name"], color=w["color"],
                                pos=w["truth"][f] if f < len(w["truth"]) else w["truth"][-1],
                                hard0=np.asarray(wd["inflated"][0]),
                                tube=safety_tube_polygon(wd["inflated"])))
        busy = [dict(hull=lk.hull, color=lk.color) for lk in sim.locks]
        snaps.append(dict(frame=f, amrs=amrs, workers=workers, busy=busy,
                          n_local=sum(1 for a in sim.amrs if a.local_mode),
                          done=sum(1 for a in sim.amrs if a.is_done()),
                          collisions=sum(1 for a in sim.amrs if a.collided),
                          min_clear=(min_clear if np.isfinite(min_clear) else 0.0)))
    return sim, snaps


def main(argv=None):
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", type=str, default="logs/V1_local/local_best.pt")
    pa.add_argument("--frames", type=int, default=420)
    pa.add_argument("--seed", type=int, default=7)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--hidden", type=int, default=64)
    pa.add_argument("--output", type=str, default="outputs/step_e_local_rl_demo.mp4")
    args = pa.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = args.model if Path(args.model).is_absolute() else str(repo / args.model)
    sim, snaps = _simulate_rl(model, args.frames, args.workers, args.amrs,
                              args.seed, args.hidden, device)
    out = Path(args.output)
    if not out.is_absolute():
        out = repo / out
    saved = animate_local(sim, snaps, len(snaps), out, False,
                          "Step E (RL): attention policy local PATH replanning + 2D shield")
    print(f"VIDEO SAVED {saved}")


if __name__ == "__main__":
    main()
