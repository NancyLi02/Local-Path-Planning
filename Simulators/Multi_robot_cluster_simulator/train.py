"""PPO trainer for the multi-robot cluster local-replanning policy (GPU).

Random scenarios every episode (1-6 AMRs, parallel/perp rails, crossing workers).
Saves the best policy + eval videos so training quality is visible.

    python -m Simulators.Multi_robot_cluster_simulator.train --timesteps 1500000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .env import MultiRobotClusterEnv, ClusterEnvConfig, OBS_DIM
from .policy import ClusterPolicy

_ROOT = Path(__file__).resolve().parents[2]
_LOG = _ROOT / "logs" / "cluster_rl"


def evaluate(env, policy, seeds, device):
    succ, comp, coll, rets = [], [], [], []
    for s in seeds:
        obs = env.reset(s); done = False; R = 0.0; info = {}
        while not done:
            mask = env.active_mask
            ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
            mt = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
            a, _, _ = policy.act(ot, mt, deterministic=True)
            obs, r, done, info = env.step(a[0].cpu().numpy()); R += r
        succ.append(info["success"]); comp.append(info["completion"])
        coll.append(1.0 if info["collision"] else 0.0); rets.append(R)
    return dict(success=np.mean(succ), completion=np.mean(comp),
                collision=np.mean(coll), ret=np.mean(rets))


def save_videos(env, policy, seeds, outdir, device, tag):
    from .render import render_episode
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    res = []
    for s in seeds:
        path = outdir / f"{tag}_seed{s}.mp4"
        _, r, info = render_episode(env, policy, s, path, device=device)
        res.append(r)
    return res


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--timesteps", type=int, default=1_500_000)
    pa.add_argument("--rollout", type=int, default=4096)
    pa.add_argument("--epochs", type=int, default=4)
    pa.add_argument("--minibatch", type=int, default=1024)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--gamma", type=float, default=0.99)
    pa.add_argument("--lam", type=float, default=0.95)
    pa.add_argument("--clip", type=float, default=0.2)
    pa.add_argument("--ent-coef", type=float, default=0.005)
    pa.add_argument("--vf-coef", type=float, default=0.5)
    pa.add_argument("--hidden", type=int, default=96)
    pa.add_argument("--eval-every", type=int, default=20480)
    pa.add_argument("--name", type=str, default="cluster")
    args = pa.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    env = MultiRobotClusterEnv(); eval_env = MultiRobotClusterEnv()
    policy = ClusterPolicy(OBS_DIM, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    _LOG.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(_LOG / "tb" / args.name))
    except Exception as e:
        print(f"  (tensorboard unavailable: {e})"); writer = None

    eval_seeds = list(range(10000, 10030))
    print(f"device={device}  obs_dim={OBS_DIM}")
    base = evaluate(eval_env, policy, eval_seeds, device)
    print(f"untrained: success={base['success']*100:.0f}%  completion={base['completion']*100:.0f}%  "
          f"collision={base['collision']*100:.0f}%  ret={base['ret']:.1f}\n")

    obs = env.reset(None); gstep = 0; best = -1e9; next_eval = args.eval_every
    while gstep < args.timesteps:
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
            obs = env.reset(None) if done else nobs
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
                ratio = torch.exp(lp - tlp[mb])
                s1 = ratio * tadv[mb]; s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * tadv[mb]
                loss = (-torch.min(s1, s2).mean() + args.vf_coef * ((vp - tret[mb]) ** 2).mean()
                        - args.ent_coef * ent.mean())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()

        ev = evaluate(eval_env, policy, eval_seeds, device)
        print(f"step {gstep:>8d}  success={ev['success']*100:5.1f}%  completion={ev['completion']*100:5.1f}%  "
              f"collision={ev['collision']*100:5.1f}%  ret={ev['ret']:7.1f}")
        if writer is not None:
            for k in ("success", "completion", "collision", "ret"):
                writer.add_scalar(f"eval/{k}", ev[k], gstep)
        score = ev["success"] - 0.5 * ev["collision"]
        if score > best:
            best = score; torch.save(policy.state_dict(), _LOG / f"{args.name}_best.pt")
        if gstep >= next_eval:
            next_eval += args.eval_every
            save_videos(eval_env, policy, [10000, 10001, 10002], _LOG / "videos", device,
                        tag=f"{args.name}_step{gstep}")

    torch.save(policy.state_dict(), _LOG / f"{args.name}_final.pt")
    print(f"\nSaved -> {_LOG/(args.name+'_final.pt')}  best score {best:.3f}")
    save_videos(eval_env, policy, list(range(10000, 10006)), _LOG / "videos", device,
                tag=f"{args.name}_final")
    print(f"eval videos -> {_LOG/'videos'}")


if __name__ == "__main__":
    main()
