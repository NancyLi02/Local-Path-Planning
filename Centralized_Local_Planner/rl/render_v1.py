"""Render the V1 demo: trained attention policy driving the fleet, behind the
shield, with the same map + Gantt layout as the V0 demo.

    python -m Centralized_Local_Planner.rl.render_v1 --model logs/V1/v1_best.pt --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .fleet_env import FleetEnv, FleetEnvConfig, OBS_DIM
from .policy import AttentionPolicy
from ..tools.geometry import safety_tube_polygon
from ..viz.render_common import STATUS_COLOR
from ..viz.render_replanning import animate_snapshots


def _label(factor):
    if factor < 0.05:
        return "STOP"
    if factor < 0.95:
        return "SLOW"
    return "GO"


def _simulate_v1(model, num_frames, num_workers, num_amrs, seed, hidden, device):
    env = FleetEnv(FleetEnvConfig(num_frames=num_frames, num_workers=num_workers,
                                  num_amrs=num_amrs))
    policy = AttentionPolicy(OBS_DIM, hidden=hidden).to(device)
    policy.load_state_dict(torch.load(model, map_location=device))
    policy.eval()

    obs = env.reset(seed)
    snaps = []
    min_clear = float("inf")
    last_gantt = {a.name: [STATUS_COLOR["PENDING"]] * env.steps for a in env.amrs}
    done = False
    for f in range(num_frames):
        if done:
            break
        mask = env.active_mask
        # Gantt colours from the threat at the planner-allowed speed (pre-step).
        gantt_now = {}
        for i, a in enumerate(env.amrs):
            if mask[i]:
                hm = env._hard_mask(i, max(env._cur[i]["base_v"], 1e-9))
                gantt_now[a.name] = [STATUS_COLOR["REPLAN"] if h else STATUS_COLOR["CLEAR"] for h in hm]
        # policy action (deterministic), then advance
        ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        mt = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
        act, _, _ = policy.act(ot, mt, deterministic=True)
        action = act[0].cpu().numpy()
        obs, r, done, info = env.step(action)

        amrs = []
        for a in env.amrs:
            pos = a.position_at(a.progress)
            spawned = a.is_spawned(f)
            if not a.collided and not a.is_done() and spawned:
                for w in env.workers:
                    wp = w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]
                    min_clear = min(min_clear, float(np.linalg.norm(pos - wp)))
            if a.collided:
                gantt = last_gantt[a.name]
            elif a.name in gantt_now:
                gantt = gantt_now[a.name]; last_gantt[a.name] = gantt
            elif not spawned:
                gantt = [STATUS_COLOR["PENDING"]] * env.steps
            elif a.is_done():
                gantt = [STATUS_COLOR["DONE"]] * env.steps
            else:
                gantt = [STATUS_COLOR["CLEAR"]] * env.steps
            factor = a.actual_speed / max(a.commanded_speed, 1e-9)
            amrs.append(dict(name=a.name, color=a.color, pos=pos,
                             heading=a.heading_at(a.progress),
                             action=_label(factor),
                             collided=a.collided, done=a.is_done(), spawned=spawned,
                             gantt=list(gantt), coll_meta=None))
        workers = []
        for w, wd in zip(env.workers, env.worker_frames[f]):
            workers.append(dict(name=w["name"], color=w["color"],
                                pos=w["truth"][f] if f < len(w["truth"]) else w["truth"][-1],
                                hard0=np.asarray(wd["inflated"][0]),
                                tube=safety_tube_polygon(wd["inflated"])))
        snaps.append(dict(amrs=amrs, workers=workers, frame=f,
                          collisions=sum(1 for a in env.amrs if a.collided),
                          done=sum(1 for a in env.amrs if a.is_done()),
                          min_clear=(min_clear if np.isfinite(min_clear) else 0.0)))
    return env, snaps


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--model", type=str, default="logs/V1/v1_best.pt")
    pa.add_argument("--seed", type=int, default=7)
    pa.add_argument("--frames", type=int, default=420)
    pa.add_argument("--workers", type=int, default=2)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--hidden", type=int, default=64)
    pa.add_argument("--output", type=str, default="outputs/step_e_v1_replanning_demo.mp4")
    args = pa.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env, snaps = _simulate_v1(repo / args.model if not Path(args.model).is_absolute() else args.model,
                              args.frames, args.workers, args.amrs, args.seed, args.hidden, device)
    amr_meta = [dict(name=a.name, color=a.color, waypoints=a.waypoints) for a in env.amrs]
    worker_meta = [dict(name=w["name"], color=w["color"]) for w in env.workers]
    out = Path(args.output)
    if not out.is_absolute():
        out = repo / out
    saved = animate_snapshots(
        amr_meta, worker_meta, snaps, env.dt, env.steps, len(snaps),
        out, preview=False,
        title="Step E (V1): attention centralized policy + space-time shield",
        shield_seconds=env.steps * env.dt)
    print(f"VIDEO SAVED {saved}")


if __name__ == "__main__":
    main()
