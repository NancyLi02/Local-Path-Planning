"""Category analysis: why critic-greedy reduces collisions but lowers success.

Read-only. No env/reward/model changes. Reruns ACTOR and CRITIC-GREEDY on the same
seeds (reward breakdown enabled), then buckets seed-matched episodes into:

  A) actor SUCCESS  & greedy FAIL      (the success regression)
  B) actor COLLISION & greedy SUCCESS  (the collisions greedy fixes)

For each category it reports trajectory metrics (min human dist, path deviation,
progress, time near humans, final lateral / recovery) and summed reward components.

Usage:
    ./venv/bin/python diagnostics/greedy_vs_actor_categories.py --name sac_v10 --episodes 60
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3 import SAC

from Simulators.Single_robot_simulator import LocalPlannerEnv
from Simulators.Single_robot_simulator.policies import obs_to_path_goal
from training_files.SAC_train import EVAL_ENV_CFG, _best_model_path, _model_path

REWARD_KEYS = ["progress", "safety", "collision", "return_path", "time",
               "deviation", "heading", "speed", "idle", "success_bonus"]


def candidate_grid(env):
    lo, hi = env.action_space.low, env.action_space.high
    fwds = [0.0, 0.75, 1.5, 2.25, 3.0]
    lats = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    return np.clip(np.array([[f, l] for f, l in product(fwds, lats)], dtype=np.float32), lo, hi)


def greedy_action(model, env, obs, actor_action, grid):
    lo, hi = env.action_space.low, env.action_space.high
    ret = np.clip(obs_to_path_goal(obs, env.cfg, lookahead_idx=3), lo, hi).astype(np.float32)
    cands = np.vstack([grid, actor_action.reshape(1, -1).astype(np.float32), ret.reshape(1, -1)])
    obs_t, _ = model.policy.obs_to_tensor(obs)
    obs_b = obs_t.repeat(cands.shape[0], 1)
    a_t = torch.as_tensor(model.policy.scale_action(cands), dtype=torch.float32, device=model.device)
    with torch.no_grad():
        qs = model.critic(obs_b, a_t)
    q = torch.stack(qs, dim=0).min(dim=0).values.squeeze(-1).cpu().numpy()
    return cands[int(np.argmax(q))].astype(np.float32)


def run_episode(model, env, seed, mode, grid, stop_speed):
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    L = env.path.total_length
    lat_scale = max(env.cfg["corridor_w"] * 1.5, env.cfg["success_lat_thresh"], 0.5)
    safety = float(env.cfg["safety_dist"])
    detect = float(env.cfg.get("human_detect_radius", 7.0))
    min_dh = float("inf")
    lats, dhs, rvs = [], [], []
    near_bubble = near_detect = enc_steps = stop_steps = 0
    rterms = {k: 0.0 for k in REWARD_KEYS}
    final_lat = 0.0

    while not done:
        actor_action, _ = model.predict(obs, deterministic=True)
        action = greedy_action(model, env, obs, actor_action, grid) if (mode == "greedy" and env._rl_active) else actor_action
        vis = bool(env._human_visible)
        if vis:
            enc_steps += 1
            if env.rv < stop_speed:
                stop_steps += 1
        obs, _, term, trunc, info = env.step(action)
        _, lat, _ = env.path.closest_point(env.rx, env.ry, s_hint=env.cur_s, search_radius=5.0)
        final_lat = abs(lat)
        lats.append(abs(lat))
        rvs.append(env.rv)
        if env._human_exists:
            dh = float(np.hypot(env.rx - env.hx, env.ry - env.hy))
            min_dh = min(min_dh, dh)
            dhs.append(dh)
            if vis and dh < safety:
                near_bubble += 1
            if vis and dh < detect:
                near_detect += 1
        rt = info.get("reward_terms", {})
        for k in REWARD_KEYS:
            rterms[k] += float(rt.get(k, 0.0))
        done = term or trunc

    tag = ("collision" if info.get("collision") else "success" if info.get("success")
           else "timeout" if info.get("timeout") else "other")
    return {
        "tag": tag, "min_dh": min_dh, "max_lat": max(lats) if lats else 0.0,
        "mean_lat": float(np.mean(lats)) if lats else 0.0, "final_lat": final_lat,
        "progress": env.cur_s / L, "steps": info.get("step", 0),
        "near_bubble": near_bubble, "near_detect": near_detect, "enc_steps": enc_steps,
        "stop_steps": stop_steps, "mean_rv": float(np.mean(rvs)) if rvs else 0.0,
        "rterms": rterms,
    }


def _fmt_cat(name, pairs):
    print("\n" + "=" * 96)
    print(f"{name}   (n={len(pairs)} seeds)")
    print("=" * 96)
    if not pairs:
        print("  (none)")
        return
    hdr = (f"{'seed':>5} {'mode':>7} {'tag':>9} {'minDH':>6} {'maxLat':>6} {'finLat':>6} "
           f"{'prog':>5} {'steps':>5} {'nBub':>5} {'nDet':>5} {'mRV':>5}")
    print(hdr); print("-" * len(hdr))
    for seed, a, g in pairs:
        for mode, r in (("actor", a), ("greedy", g)):
            print(f"{seed:>5} {mode:>7} {r['tag']:>9} {r['min_dh']:6.2f} {r['max_lat']:6.2f} "
                  f"{r['final_lat']:6.2f} {r['progress']:5.2f} {r['steps']:5d} "
                  f"{r['near_bubble']:5d} {r['near_detect']:5d} {r['mean_rv']:5.2f}")

    def avg(rows, key):
        return float(np.mean([r[key] for r in rows]))
    A = [a for _, a, _ in pairs]
    G = [g for _, _, g in pairs]
    print("-" * len(hdr))
    for key, lab in [("min_dh", "min human dist"), ("max_lat", "max path dev"),
                     ("final_lat", "final |lat| (recovery)"), ("progress", "goal progress"),
                     ("steps", "steps"), ("near_bubble", "steps in bubble"),
                     ("near_detect", "steps near human"), ("mean_rv", "mean speed")]:
        print(f"  {lab:<26} actor={avg(A,key):8.3f}   greedy={avg(G,key):8.3f}")

    print(f"\n  Reward components (episode sums, mean over category):")
    print(f"  {'component':<16}{'ACTOR':>12}{'GREEDY':>12}{'delta(G-A)':>12}")
    for k in REWARD_KEYS:
        va = float(np.mean([r["rterms"][k] for r in A]))
        vg = float(np.mean([r["rterms"][k] for r in G]))
        print(f"  {k:<16}{va:>12.2f}{vg:>12.2f}{vg-va:>12.2f}")


def run(args):
    load_stem = _best_model_path(args.name) if args.use_best else _model_path(args.name)
    if not Path(str(load_stem) + ".zip").exists():
        raise FileNotFoundError(f"Model not found: {load_stem}.zip")
    model = SAC.load(str(load_stem))
    cfg = dict(EVAL_ENV_CFG)
    cfg["return_reward_breakdown"] = True
    env = LocalPlannerEnv(config=cfg)
    grid = candidate_grid(env)

    actor, greedy = {}, {}
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        actor[seed] = run_episode(model, env, seed, "actor", grid, args.stop_speed)
        greedy[seed] = run_episode(model, env, seed, "greedy", grid, args.stop_speed)
    env.close()

    catA = [(s, actor[s], greedy[s]) for s in actor
            if actor[s]["tag"] == "success" and greedy[s]["tag"] != "success"]
    catB = [(s, actor[s], greedy[s]) for s in actor
            if actor[s]["tag"] == "collision" and greedy[s]["tag"] == "success"]

    print(f"\n########## CATEGORY ANALYSIS: {args.name} ({args.episodes} eps) ##########")
    _fmt_cat("A) ACTOR SUCCEEDS but CRITIC-GREEDY FAILS", catA)
    _fmt_cat("B) ACTOR COLLIDES but CRITIC-GREEDY SUCCEEDS", catB)

    # greedy failure-mode breakdown for category A
    if catA:
        G = [g for _, _, g in catA]
        timeouts = sum(g["tag"] == "timeout" for g in G)
        big_dev = sum(g["max_lat"] > 1.5 for g in G)
        poor_recover = sum(g["final_lat"] > 0.3 for g in G)
        low_prog = sum(g["progress"] < 0.9 for g in G)
        over_avoid = sum(g["min_dh"] > a["min_dh"] + 0.3 for _, a, g in catA)
        print("\n  Category-A greedy failure modes:")
        print(f"    timeouts:                     {timeouts}/{len(G)}")
        print(f"    over-avoids (minDH >> actor): {over_avoid}/{len(G)}")
        print(f"    leaves path far (maxLat>1.5): {big_dev}/{len(G)}")
        print(f"    fails to recover (finLat>0.3):{poor_recover}/{len(G)}")
        print(f"    low progress (<0.9):          {low_prog}/{len(G)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="sac_v10")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed0", type=int, default=2000)
    p.add_argument("--use-best", action="store_true", default=True)
    p.add_argument("--stop-speed", type=float, default=0.15)
    run(p.parse_args())
