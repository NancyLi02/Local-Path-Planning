"""PPO trainer for the spatial local-replanning attention policy (Phase 2, GPU).

    python -m Centralized_Local_Planner.rl.train_local --timesteps 200000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .local_cluster_env import LocalClusterEnv, LocalEnvConfig, OBS_DIM
from .local_policy import LocalAttentionPolicy

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "V1_local"


def _greedy_action(obs, mask):
    """Baseline: head straight to the exit (to-exit dir = obs[:,0:2])."""
    a = np.zeros((obs.shape[0], 2), dtype=np.float32)
    for i in range(obs.shape[0]):
        if mask[i]:
            v = obs[i, 0:2]
            n = float(np.linalg.norm(v))
            a[i] = v / n if n > 1e-6 else 0.0
    return a


def behavior_clone(policy, env, seeds, device, epochs=400, mb=512):
    """Warm-start: imitate the greedy 'head-to-exit' controller so PPO starts
    from a competent policy instead of random wandering."""
    O, M, A = [], [], []
    for s in seeds:
        obs = env.reset(s); done = False
        while not done:
            mask = env.active_mask
            act = _greedy_action(obs, mask)
            if mask.any():
                O.append(obs.copy()); M.append(mask.copy()); A.append(act)
            obs, _, done, _ = env.step(act)
    if not O:
        return
    O = torch.as_tensor(np.array(O), dtype=torch.float32, device=device)
    M = torch.as_tensor(np.array(M), dtype=torch.bool, device=device)
    A = torch.as_tensor(np.array(A), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    n = len(O); idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        last = 0.0
        for st in range(0, n, mb):
            b = idx[st:st + mb]
            mean, _, _ = policy.forward(O[b], M[b])
            pred = torch.tanh(mean)
            mf = M[b].unsqueeze(-1).float()
            loss = (((pred - A[b]) ** 2) * mf).sum() / mf.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step(); last = float(loss.item())
    print(f"  BC warm-start done ({n} samples), final imitation loss {last:.4f}")


def evaluate(env, policy, seeds, device, mode="policy"):
    comp, coll, prog, rets = [], [], [], []
    for s in seeds:
        obs = env.reset(s); R = 0.0; info = {}; done = False
        while not done:
            mask = env.active_mask
            if mode == "greedy":
                act = _greedy_action(obs, mask)
            else:
                ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                mt = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
                a, _, _ = policy.act(ot, mt, deterministic=True)
                act = a[0].cpu().numpy()
            obs, r, done, info = env.step(act); R += r
        comp.append(info["completion"]); coll.append(info["collided"])
        prog.append(info["mean_progress"]); rets.append(R)
    return dict(completion=np.mean(comp), collided=np.mean(coll),
                progress=np.mean(prog), ret=np.mean(rets))


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--timesteps", type=int, default=200_000)
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
    pa.add_argument("--frames", type=int, default=420)
    pa.add_argument("--name", type=str, default="local")
    pa.add_argument("--init-model", type=str, default="",
                    help="checkpoint to fine-tune from (e.g. the episodic policy)")
    pa.add_argument("--critic-warmup", type=int, default=0,
                    help="rollout iters that update ONLY the value head first")
    args = pa.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    seeds = list(range(args.train_seeds))
    env = LocalClusterEnv(LocalEnvConfig(num_frames=args.frames))
    eval_env = LocalClusterEnv(LocalEnvConfig(num_frames=args.frames))
    policy = LocalAttentionPolicy(OBS_DIM, hidden=args.hidden).to(device)
    if args.init_model:
        repo = Path(__file__).resolve().parents[2]
        mp = args.init_model if Path(args.init_model).is_absolute() else str(repo / args.init_model)
        policy.load_state_dict(torch.load(mp, map_location=device))
        print(f"  initialised from {args.init_model}")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(_LOG_DIR / "tb" / args.name))
    except Exception as e:                                 # pragma: no cover
        print(f"  (tensorboard unavailable: {e}; logging to stdout only)")
        writer = None

    print(f"device={device} seeds={seeds}  warming worker caches ...")
    t0 = time.time()
    for s in seeds:
        env.reset(s)
    print(f"  ready in {time.time()-t0:.1f}s")
    base = evaluate(eval_env, policy, seeds, device, mode="greedy")
    print(f"greedy baseline: completion={base['completion']*100:.1f}%  "
          f"collided={base['collided']:.2f}  ret={base['ret']:.1f}")
    init = evaluate(eval_env, policy, seeds, device)   # untrained direct policy
    print(f"init (untrained): completion={init['completion']*100:.1f}%  "
          f"collided={init['collided']:.2f}  ret={init['ret']:.1f}\n")

    obs = env.reset(seeds[0]); si = 0; gstep = 0; best = init["ret"]; it = 0
    if args.init_model:
        torch.save(policy.state_dict(), _LOG_DIR / f"{args.name}_best.pt")  # floor = init
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
                si = (si + 1) % len(seeds); obs = env.reset(seeds[si])
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
                    loss = v_loss
                else:
                    ratio = torch.exp(lp - tlp[mb])
                    s1 = ratio * tadv[mb]
                    s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * tadv[mb]
                    loss = (-torch.min(s1, s2).mean() + args.vf_coef * v_loss
                            - args.ent_coef * ent.mean())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()

        ev = evaluate(eval_env, policy, seeds, device)
        print(f"step {gstep:>7d}  ret={ev['ret']:7.1f}  completion={ev['completion']*100:5.1f}%  "
              f"progress={ev['progress']*100:5.1f}%  collided={ev['collided']:.2f}  "
              f"(greedy ret={base['ret']:.1f})")
        if writer is not None:
            writer.add_scalar("eval/return", ev["ret"], gstep)
            writer.add_scalar("eval/completion", ev["completion"], gstep)
            writer.add_scalar("eval/progress", ev["progress"], gstep)
            writer.add_scalar("eval/collided", ev["collided"], gstep)
            writer.add_scalar("ref/greedy_return", base["ret"], gstep)
        if ev["ret"] > best:
            best = ev["ret"]; torch.save(policy.state_dict(), _LOG_DIR / f"{args.name}_best.pt")

    torch.save(policy.state_dict(), _LOG_DIR / f"{args.name}_final.pt")
    print(f"\nSaved -> {_LOG_DIR / (args.name + '_final.pt')}  best ret {best:.1f} (greedy {base['ret']:.1f})")


if __name__ == "__main__":
    main()
