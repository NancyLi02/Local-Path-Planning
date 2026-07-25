"""PPO trainer for the V1 attention fleet policy (PyTorch, GPU).

Compact single-env PPO with GAE. The action (per-AMR speed factor) is projected
through the V0 shield inside FleetEnv, so training only optimizes efficiency.
Baseline for comparison is "always factor 1" (== V0) evaluated in the same env.

    python -m Centralized_Local_Planner.rl.train_v1 --timesteps 150000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .fleet_env import FleetEnv, FleetEnvConfig, OBS_DIM
from .policy import AttentionPolicy

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "V1"


def evaluate(env, policy, seeds, device, deterministic=True, v0=False):
    """Run full episodes; return mean metrics. v0=True -> always factor 1."""
    comp, prog, coll, stops, rets = [], [], [], [], []
    for s in seeds:
        obs = env.reset(s)
        R = 0.0; info = {}
        done = False
        while not done:
            mask = env.active_mask
            if v0:
                act = np.ones(env.N, dtype=np.float32)
            else:
                ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                mt = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
                a, _, _ = policy.act(ot, mt, deterministic=deterministic)
                act = a[0].cpu().numpy()
            obs, r, done, info = env.step(act)
            R += r
        comp.append(info["completion"]); prog.append(info["progress"])
        coll.append(info["collided"]); stops.append(info["n_stop"]); rets.append(R)
    return dict(completion=np.mean(comp), progress=np.mean(prog),
                collided=np.mean(coll), ret=np.mean(rets))


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--timesteps", type=int, default=150_000)
    pa.add_argument("--rollout", type=int, default=2048)
    pa.add_argument("--epochs", type=int, default=4)
    pa.add_argument("--minibatch", type=int, default=512)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--gamma", type=float, default=0.99)
    pa.add_argument("--lam", type=float, default=0.95)
    pa.add_argument("--clip", type=float, default=0.2)
    pa.add_argument("--ent-coef", type=float, default=0.01)
    pa.add_argument("--vf-coef", type=float, default=0.5)
    pa.add_argument("--hidden", type=int, default=64)
    pa.add_argument("--train-seeds", type=int, default=4)
    pa.add_argument("--frames", type=int, default=360)
    pa.add_argument("--name", type=str, default="v1")
    args = pa.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    train_seeds = list(range(args.train_seeds))

    env = FleetEnv(FleetEnvConfig(num_frames=args.frames))
    eval_env = FleetEnv(FleetEnvConfig(num_frames=args.frames))   # separate: keeps training env state intact
    policy = AttentionPolicy(OBS_DIM, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"device={device}  train_seeds={train_seeds}  warming up no-go caches ...")
    t0 = time.time()
    for s in train_seeds:
        env.reset(s)
    print(f"  caches ready in {time.time()-t0:.1f}s")

    # baseline (V0) reference in this env
    base = evaluate(eval_env, policy, train_seeds, device, v0=True)
    print(f"V0 baseline (in-env): completion={base['completion']*100:.1f}%  "
          f"progress={base['progress']*100:.1f}%  collided={base['collided']:.2f}  ret={base['ret']:.1f}\n")

    # ---- rollout state ----
    seed_iter = 0
    obs = env.reset(train_seeds[seed_iter])
    global_step = 0
    best_ret = -1e9
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(_LOG_DIR / "tb" / args.name))
    except Exception as e:                                 # pragma: no cover
        print(f"  (tensorboard unavailable: {e})"); writer = None

    while global_step < args.timesteps:
        # ---------- collect rollout ----------
        b_obs, b_mask, b_act, b_logp, b_val, b_rew, b_done = [], [], [], [], [], [], []
        for _ in range(args.rollout):
            mask = env.active_mask
            ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
            mt = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
            with torch.no_grad():
                act, logp, val = policy.act(ot, mt, deterministic=False)
            a = act[0].cpu().numpy()
            nobs, r, done, info = env.step(a)
            b_obs.append(obs); b_mask.append(mask); b_act.append(a)
            b_logp.append(float(logp.item())); b_val.append(float(val.item()))
            b_rew.append(r); b_done.append(done)
            global_step += 1
            if done:
                seed_iter = (seed_iter + 1) % len(train_seeds)
                obs = env.reset(train_seeds[seed_iter])
            else:
                obs = nobs

        # bootstrap value
        with torch.no_grad():
            mt = torch.as_tensor(env.active_mask[None], dtype=torch.bool, device=device)
            ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
            _, _, last_val = policy.act(ot, mt)
            last_val = float(last_val.item())

        # ---------- GAE ----------
        rew = np.array(b_rew, dtype=np.float32)
        val = np.array(b_val, dtype=np.float32)
        done = np.array(b_done, dtype=np.float32)
        adv = np.zeros_like(rew); gae = 0.0
        for t in reversed(range(len(rew))):
            next_val = last_val if t == len(rew) - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + args.gamma * next_val * nonterm - val[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        t_obs = torch.as_tensor(np.array(b_obs), dtype=torch.float32, device=device)
        t_mask = torch.as_tensor(np.array(b_mask), dtype=torch.bool, device=device)
        t_act = torch.as_tensor(np.array(b_act), dtype=torch.float32, device=device)
        t_logp = torch.as_tensor(np.array(b_logp), dtype=torch.float32, device=device)
        t_adv = torch.as_tensor(adv, dtype=torch.float32, device=device)
        t_ret = torch.as_tensor(ret, dtype=torch.float32, device=device)

        # ---------- PPO update ----------
        n = len(b_rew); idx = np.arange(n)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, args.minibatch):
                mb = idx[start:start + args.minibatch]
                logp, ent, val_pred = policy.evaluate(t_obs[mb], t_mask[mb], t_act[mb])
                ratio = torch.exp(logp - t_logp[mb])
                s1 = ratio * t_adv[mb]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * t_adv[mb]
                pg_loss = -torch.min(s1, s2).mean()
                v_loss = ((val_pred - t_ret[mb]) ** 2).mean()
                ent_loss = -ent.mean()
                loss = pg_loss + args.vf_coef * v_loss + args.ent_coef * ent_loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        # ---------- periodic eval ----------
        ev = evaluate(eval_env, policy, train_seeds, device, deterministic=True)
        print(f"step {global_step:>7d}  ret={ev['ret']:7.1f}  "
              f"completion={ev['completion']*100:5.1f}%  progress={ev['progress']*100:5.1f}%  "
              f"collided={ev['collided']:.2f}  (V0 ret={base['ret']:.1f})")
        if writer is not None:
            writer.add_scalar("eval/return", ev["ret"], global_step)
            writer.add_scalar("eval/completion", ev["completion"], global_step)
            writer.add_scalar("eval/progress", ev["progress"], global_step)
            writer.add_scalar("eval/collided", ev["collided"], global_step)
            writer.add_scalar("ref/v0_return", base["ret"], global_step)
        if ev["ret"] > best_ret:
            best_ret = ev["ret"]
            torch.save(policy.state_dict(), _LOG_DIR / f"{args.name}_best.pt")

    torch.save(policy.state_dict(), _LOG_DIR / f"{args.name}_final.pt")
    print(f"\nSaved model -> {_LOG_DIR / (args.name + '_final.pt')}")
    print(f"Best eval return {best_ret:.1f}  (V0 baseline {base['ret']:.1f})")


if __name__ == "__main__":
    main()
