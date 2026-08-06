"""PPO on the per-cluster EPISODIC env (dense rejoin signal -> learnable from
scratch). Trained policy transfers to the full simulator (same 12-dim obs).

    python -m Centralized_Local_Planner.rl.train_episodic --timesteps 300000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .episodic_env import EpisodicLocalEnv, EpisodicConfig, OBS_DIM
from .local_policy import LocalAttentionPolicy

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "V1_local"


def _greedy_action(obs, mask):
    a = np.zeros((obs.shape[0], 2), dtype=np.float32)
    for i in range(obs.shape[0]):
        if mask[i]:
            v = obs[i, 0:2]; n = float(np.linalg.norm(v))
            a[i] = v / n if n > 1e-6 else 0.0
    return a


def _rule_action(obs, mask):
    """Evade-aware 'rule' direction computed from obs (matches RuleLocalReplanner):
    head to exit, but blend in 'away from nearest worker' when close. Used as the
    BC target so the policy inherits the robust avoidance behaviour."""
    a = np.zeros((obs.shape[0], 2), dtype=np.float32)
    for i in range(obs.shape[0]):
        if not mask[i]:
            continue
        g = obs[i, 0:2]; gn = float(np.linalg.norm(g))
        gdir = g / gn if gn > 1e-6 else np.zeros(2)
        rw = obs[i, 3:5]; rwn = float(np.linalg.norm(rw))
        dw_norm = obs[i, 5]                      # = dist_to_worker / map_diag
        if dw_norm < 0.064 and rwn > 1e-6:       # worker within ~1.5 m
            away = -rw / rwn
            pref = gdir * 0.4 + away
            pn = float(np.linalg.norm(pref))
            a[i] = pref / pn if pn > 1e-6 else gdir
        else:
            a[i] = gdir
    return a


def behavior_clone(policy, env, device, epochs=600, mb=512):
    """Warm-start the DIRECT policy by imitating greedy 'head-to-exit' over all
    harvested scenarios. The action stays fully policy-output (not a residual);
    BC just makes it competent so PPO improves from a working start instead of
    stalling. PPO then trains on the same env, correcting any imitation drift."""
    O, M, A = [], [], []
    for sel in range(env.n_scenarios()):
        obs = env.reset(sel); done = False
        while not done:
            mask = env.active_mask
            act = _greedy_action(obs, mask)      # imitate straight-to-exit; shield = safety
            if mask.any():
                O.append(obs.copy()); M.append(mask.copy()); A.append(act)
            obs, _, done, _ = env.step(act)
    O = torch.as_tensor(np.array(O), dtype=torch.float32, device=device)
    M = torch.as_tensor(np.array(M), dtype=torch.bool, device=device)
    A = torch.as_tensor(np.array(A), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    n = len(O); idx = np.arange(n)
    last = 0.0
    for _ in range(epochs):
        np.random.shuffle(idx)
        for st in range(0, n, mb):
            b = idx[st:st + mb]
            mean, _, _ = policy.forward(O[b], M[b])
            pred = torch.tanh(mean)
            mf = M[b].unsqueeze(-1).float()
            loss = (((pred - A[b]) ** 2) * mf).sum() / mf.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step(); last = float(loss.item())
    print(f"  BC warm-start: {n} samples, imitation loss {last:.4f}")


def evaluate(env, policy, n_eval, device, mode="policy"):
    comp, coll, rets = [], [], []
    for sel in range(n_eval):
        obs = env.reset(sel); R = 0.0; info = {}; done = False
        while not done:
            mask = env.active_mask
            if mode == "greedy":
                act = _greedy_action(obs, mask)
            else:
                ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                mt = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
                a, _, _ = policy.act(ot, mt, deterministic=True); act = a[0].cpu().numpy()
            obs, r, done, info = env.step(act); R += r
        comp.append(info["completion"]); coll.append(info["collided"]); rets.append(R)
    return dict(completion=np.mean(comp), collided=np.mean(coll), ret=np.mean(rets))


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--timesteps", type=int, default=300_000)
    pa.add_argument("--rollout", type=int, default=2048)
    pa.add_argument("--epochs", type=int, default=4)
    pa.add_argument("--minibatch", type=int, default=512)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--gamma", type=float, default=0.99)
    pa.add_argument("--lam", type=float, default=0.95)
    pa.add_argument("--clip", type=float, default=0.2)
    pa.add_argument("--ent-coef", type=float, default=0.01)
    pa.add_argument("--vf-coef", type=float, default=0.5)
    pa.add_argument("--critic-warmup", type=int, default=8,
                    help="rollout iters that update ONLY the value head (so the "
                         "cold critic does not wreck the BC-warm-started policy)")
    pa.add_argument("--hidden", type=int, default=64)
    pa.add_argument("--harvest-seeds", type=int, default=8)
    pa.add_argument("--name", type=str, default="episodic")
    args = pa.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"device={device}  harvesting cluster scenarios (seeds 0..{args.harvest_seeds-1}) ...")
    t0 = time.time()
    env = EpisodicLocalEnv(EpisodicConfig(), seeds=range(args.harvest_seeds))
    eval_env = EpisodicLocalEnv(EpisodicConfig(), seeds=range(args.harvest_seeds))
    n_eval = min(40, env.n_scenarios())
    print(f"  {env.n_scenarios()} cluster scenarios harvested in {time.time()-t0:.1f}s")

    policy = LocalAttentionPolicy(OBS_DIM, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(_LOG_DIR / "tb" / args.name))
    except Exception as e:
        print(f"  (tensorboard unavailable: {e})"); writer = None

    base = evaluate(eval_env, policy, n_eval, device, mode="greedy")
    print(f"greedy baseline: completion={base['completion']*100:.1f}%  "
          f"collided={base['collided']:.2f}  ret={base['ret']:.1f}")
    behavior_clone(policy, env, device)
    bc = evaluate(eval_env, policy, n_eval, device)
    print(f"after BC:        completion={bc['completion']*100:.1f}%  "
          f"collided={bc['collided']:.2f}  ret={bc['ret']:.1f}\n")
    torch.save(policy.state_dict(), _LOG_DIR / f"{args.name}_bc.pt")

    ep_sel = 0
    obs = env.reset(ep_sel); gstep = 0; best = bc["ret"]; it = 0
    torch.save(policy.state_dict(), _LOG_DIR / f"{args.name}_best.pt")  # BC is the floor
    while gstep < args.timesteps:
        it += 1
        critic_only = it <= args.critic_warmup
        B = dict(o=[], m=[], a=[], lp=[], v=[], r=[], d=[])
        for _ in range(args.rollout):
            mask = env.active_mask
            ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
            mt = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
            with torch.no_grad():
                act, lp, val = policy.act(ot, mt)
            a = act[0].cpu().numpy()
            nobs, r, done, info = env.step(a)
            B["o"].append(obs); B["m"].append(mask); B["a"].append(a)
            B["lp"].append(float(lp.item())); B["v"].append(float(val.item()))
            B["r"].append(r); B["d"].append(done); gstep += 1
            if done:
                ep_sel += 1; obs = env.reset(ep_sel)
            else:
                obs = nobs
        with torch.no_grad():
            mt = torch.as_tensor(env.active_mask[None], dtype=torch.bool, device=device)
            ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
            _, _, lastv = policy.act(ot, mt); lastv = float(lastv.item())

        rew = np.array(B["r"], np.float32); val = np.array(B["v"], np.float32)
        dn = np.array(B["d"], np.float32); adv = np.zeros_like(rew); g = 0.0
        for t in reversed(range(len(rew))):
            nv = lastv if t == len(rew) - 1 else val[t + 1]
            nt = 1.0 - dn[t]
            delta = rew[t] + args.gamma * nv * nt - val[t]
            g = delta + args.gamma * args.lam * nt * g; adv[t] = g
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        to = torch.as_tensor(np.array(B["o"]), dtype=torch.float32, device=device)
        tm = torch.as_tensor(np.array(B["m"]), dtype=torch.bool, device=device)
        ta = torch.as_tensor(np.array(B["a"]), dtype=torch.float32, device=device)
        tlp = torch.as_tensor(np.array(B["lp"]), dtype=torch.float32, device=device)
        tadv = torch.as_tensor(adv, dtype=torch.float32, device=device)
        tret = torch.as_tensor(ret, dtype=torch.float32, device=device)
        n = len(rew); idx = np.arange(n)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for st in range(0, n, args.minibatch):
                mb = idx[st:st + args.minibatch]
                lp, ent, vp = policy.evaluate(to[mb], tm[mb], ta[mb])
                v_loss = ((vp - tret[mb]) ** 2).mean()
                if critic_only:
                    loss = v_loss                      # warm the critic, freeze policy
                else:
                    ratio = torch.exp(lp - tlp[mb])
                    s1 = ratio * tadv[mb]
                    s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * tadv[mb]
                    loss = (-torch.min(s1, s2).mean() + args.vf_coef * v_loss
                            - args.ent_coef * ent.mean())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()

        ev = evaluate(eval_env, policy, n_eval, device)
        print(f"step {gstep:>7d}  ret={ev['ret']:7.1f}  completion={ev['completion']*100:5.1f}%  "
              f"collided={ev['collided']:.2f}  (greedy ret={base['ret']:.1f} comp={base['completion']*100:.0f}%)")
        if writer is not None:
            writer.add_scalar("eval/return", ev["ret"], gstep)
            writer.add_scalar("eval/completion", ev["completion"], gstep)
            writer.add_scalar("eval/collided", ev["collided"], gstep)
            writer.add_scalar("ref/greedy_return", base["ret"], gstep)
        if ev["ret"] > best:
            best = ev["ret"]; torch.save(policy.state_dict(), _LOG_DIR / f"{args.name}_best.pt")

    torch.save(policy.state_dict(), _LOG_DIR / f"{args.name}_final.pt")
    print(f"\nSaved -> {_LOG_DIR/(args.name+'_final.pt')}  best ret {best:.1f} "
          f"(greedy {base['ret']:.1f})")


if __name__ == "__main__":
    main()
