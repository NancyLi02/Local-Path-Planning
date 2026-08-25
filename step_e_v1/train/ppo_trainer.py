"""PPO trainer for the V1 attention policy (shared actor + centralized critic).

    python -m step_e_v1.train.ppo_trainer --timesteps 120000

Training follows the specification's plan:

    Stage 1  N = 1        behaviour-clone the V0 proposal (a competent start)
    Stage 2  N <= 2       learn yielding, slowing, small lateral shifts
    Stage 3  N <= 3       group coordination
    Stage 4  N <= 4       full conflict clusters
    Stage 5  mixed N      all harvested clusters

The critic is warmed up alone for a few iterations first: a cold value head
would otherwise inject large advantage noise and destroy the cloned policy.
Everything the policy proposes is projected through the same safety shield as
V0, so training can never trade safety for reward -- only efficiency.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..attention_policy import MultiAMRAttentionPolicy
from ..action_decoder import decode_action, encode_action_np
from ..config import V1Config
from ..env import ClusterEnv, EnvConfig
from ..v0_planner import V0SequentialReplanner

_REPO = Path(__file__).resolve().parents[2]
_LOG = _REPO / "logs" / "step_e_v1"


# ---------------------------------------------------------------------------
# Teacher (Stage 1 behaviour cloning target)
# ---------------------------------------------------------------------------

_TEACHER: dict = {}


def v0_raw_actions(env, cfg) -> np.ndarray:
    """V0's EXECUTED action for the env's current cluster, in RAW space.

    The teacher is V0's committed command, not its raw proposal: most of V0's
    competence comes from the shield's cost ranking over the candidate set, and
    V1 executes its own proposal whenever that proposal is safe. Cloning the
    proposal alone would hand V1 a strictly weaker controller than V0.
    """
    teacher = _TEACHER.get(id(cfg))
    if teacher is None:
        teacher = V0SequentialReplanner(cfg)
        _TEACHER[id(cfg)] = teacher
    raw = np.zeros((cfg.max_agents, cfg.action_dim), dtype=np.float32)
    dec = np.zeros((cfg.max_agents, cfg.action_dim), dtype=np.float32)
    if not env._agents:
        return raw, dec
    teacher.plan(env._agents, env.rt.last_worker_predictions, env.rt.map_data,
                 dt=cfg.dt)
    executed = teacher.stats.executed
    for k, amr_id in enumerate(env._ids):
        act = np.asarray(executed[amr_id], dtype=float).copy()
        act[2] = max(act[2], 0.0)      # the policy cannot command reverse
        dec[k] = act
        raw[k] = encode_action_np(act, cfg.max_forward_dist, cfg.max_lateral_offset)
    return raw, dec


def behaviour_clone(policy, env, indices, device, cfg, epochs: int = 300,
                    minibatch: int = 256) -> float:
    """Clone V0's executed command.

    The loss is taken in DECODED action space, not in the raw pre-sigmoid
    space: V0 commits saturated actions (goal_fwd = 3.0 m, speed_scale = 1.0),
    whose logits are +-9, so a raw-space MSE is dominated by how deep into the
    saturation the network sits and barely constrains the behaviour at all.
    """
    obs_buf, mask_buf, act_buf = [], [], []
    for idx in indices:
        obs = env.reset(idx)
        done = False
        while not done:
            mask = env.active_mask
            raw, dec = v0_raw_actions(env, cfg)
            if mask.any():
                obs_buf.append(obs.copy()); mask_buf.append(mask.copy()); act_buf.append(dec)
            obs, _, done, _ = env.step(raw)
    if not obs_buf:
        return float("nan")
    O = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device)
    M = torch.as_tensor(np.array(mask_buf), dtype=torch.float32, device=device)
    A = torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=device)
    scale = torch.tensor([cfg.max_forward_dist, cfg.max_lateral_offset, 1.0],
                         device=device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    n = len(O); idxs = np.arange(n); loss = float("nan")
    policy.train()
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for st in range(0, n, minibatch):
            b = idxs[st:st + minibatch]
            raw_mean, _, _ = policy.raw_forward(O[b], M[b])
            pred = decode_action(raw_mean, cfg.max_forward_dist, cfg.max_lateral_offset)
            w = M[b].unsqueeze(-1)
            l = ((((pred - A[b]) / scale) ** 2) * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step(); loss = float(l.item())
    return loss


# ---------------------------------------------------------------------------
# Evaluation on the episodic env
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(env, policy, indices, device, mode: str = "policy", cfg=None) -> dict:
    rets, res, coll, shield, stop = [], [], [], [], []
    policy.eval()
    for idx in indices:
        obs = env.reset(idx); done = False; R = 0.0; info = {}
        sh, sp, k = 0.0, 0.0, 0
        while not done:
            mask = env.active_mask
            if mode == "v0":
                raw, _ = v0_raw_actions(env, cfg)
            else:
                o = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                m = torch.as_tensor(mask[None].astype(np.float32), device=device)
                _, raw_t, _, _ = policy.act(o, m, deterministic=True)
                raw = raw_t[0].cpu().numpy()
            obs, r, done, info = env.step(raw)
            R += r; sh += info["shield_rate"]; sp += info["stop_rate"]; k += 1
        rets.append(R); res.append(info["resolved"]); coll.append(info["collisions"])
        shield.append(sh / max(k, 1)); stop.append(sp / max(k, 1))
    return dict(ret=float(np.mean(rets)), resolved=float(np.mean(res)),
                collisions=float(np.mean(coll)), shield=float(np.mean(shield)),
                stop=float(np.mean(stop)))


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--timesteps", type=int, default=120_000)
    pa.add_argument("--rollout", type=int, default=1024)
    pa.add_argument("--epochs", type=int, default=4)
    pa.add_argument("--minibatch", type=int, default=256)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--gamma", type=float, default=0.99)
    pa.add_argument("--lam", type=float, default=0.95)
    pa.add_argument("--clip", type=float, default=0.2)
    pa.add_argument("--ent-coef", type=float, default=0.0)
    pa.add_argument("--vf-coef", type=float, default=0.5)
    pa.add_argument("--critic-warmup", type=int, default=4)
    pa.add_argument("--bc-coef", type=float, default=0.5,
                    help="behaviour-anchor weight: keeps PPO near the cloned "
                         "policy. Unconstrained PPO drifts off the clone and "
                         "loses to V0 (see logs/step_e_v1/vanilla_ppo_run.log)")
    pa.add_argument("--bc-epochs", type=int, default=300)
    pa.add_argument("--bc-all", action="store_true", default=True,
                    help="clone from every harvested cluster (default)")
    pa.add_argument("--eval-every", type=int, default=1,
                    help="run the evaluation every k PPO iterations")
    pa.add_argument("--eval-clusters", type=int, default=8)
    pa.add_argument("--train-seeds", type=int, default=4)
    pa.add_argument("--frames", type=int, default=420)
    pa.add_argument("--amrs", type=int, default=6)
    pa.add_argument("--curriculum", type=str, default="1,2,3,0",
                    help="max WORKER COUNT per stage (scene difficulty); 0 = all. "
                         "The specification stages by cluster size N, but Step D "
                         "produces N=2 clusters in this factory scene, so the "
                         "difficulty that actually varies is the worker count.")
    pa.add_argument("--name", type=str, default="v1")
    pa.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = pa.parse_args(argv)

    torch.manual_seed(0); np.random.seed(0)
    cfg = V1Config(); cfg.validate()
    device = args.device
    _LOG.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  harvesting cluster snapshots ...")
    t0 = time.time()
    env = ClusterEnv(cfg, EnvConfig(frames=args.frames, num_amrs=args.amrs),
                     seeds=range(args.train_seeds))
    sizes = np.array([len(s["members"]) for s in env.snapshots])
    diff = np.array([s.get("workers", 2) for s in env.snapshots])
    print(f"  {env.n_scenarios()} clusters in {time.time()-t0:.1f}s  "
          f"sizes: {dict(zip(*np.unique(sizes, return_counts=True)))}  "
          f"workers: {dict(zip(*np.unique(diff, return_counts=True)))}")

    stages = [int(x) for x in args.curriculum.split(",")]
    all_idx = list(range(env.n_scenarios()))
    eval_idx = all_idx[:: max(len(all_idx) // max(args.eval_clusters, 1), 1)]
    eval_idx = eval_idx[: args.eval_clusters]

    policy = MultiAMRAttentionPolicy(cfg).to(device)
    base = evaluate(env, policy, eval_idx, device, mode="v0", cfg=cfg)
    print(f"V0 teacher: ret={base['ret']:7.1f} resolved={base['resolved']*100:5.1f}% "
          f"coll={base['collisions']:.2f} shield={base['shield']*100:.0f}% stop={base['stop']*100:.0f}%")

    # Only 20 clusters exist in this scene, so behaviour cloning uses all of
    # them; the curriculum then re-visits them by difficulty during PPO.
    bc_idx = all_idx if args.bc_all else (
        [i for i in all_idx if diff[i] <= max(stages[0], 1)] or all_idx)
    loss = behaviour_clone(policy, env, bc_idx, device, cfg, epochs=args.bc_epochs)
    print(f"stage 1 BC on {len(bc_idx)} clusters (workers<={max(stages[0],1)}): "
          f"loss {loss:.4f}")
    after_bc = evaluate(env, policy, eval_idx, device, cfg=cfg)
    print(f"after BC:   ret={after_bc['ret']:7.1f} resolved={after_bc['resolved']*100:5.1f}% "
          f"coll={after_bc['collisions']:.2f} shield={after_bc['shield']*100:.0f}% stop={after_bc['stop']*100:.0f}%\n")
    torch.save(dict(policy=policy.state_dict(), config=vars(cfg)), _LOG / f"{args.name}_bc.pt")

    # Frozen behaviour anchor (the cloned policy) for the PPO regulariser.
    anchor = MultiAMRAttentionPolicy(cfg).to(device)
    anchor.load_state_dict(policy.state_dict())
    anchor.eval()
    for prm in anchor.parameters():
        prm.requires_grad_(False)
    act_scale = torch.tensor([cfg.max_forward_dist, cfg.max_lateral_offset, 1.0],
                             device=device)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(_LOG / "tb" / args.name))
    except Exception as e:
        print(f"  (tensorboard unavailable: {e})"); writer = None

    best = after_bc["ret"]
    torch.save(dict(policy=policy.state_dict(), config=vars(cfg)), _LOG / f"{args.name}_best.pt")
    history = [dict(step=0, stage=0, **after_bc)]

    gstep = 0
    per_stage = max(args.timesteps // max(len(stages) - 1, 1), 1)
    ep = 0
    for stage_i, max_n in enumerate(stages[1:], start=2):
        idx_pool = [i for i in all_idx if (max_n == 0 or diff[i] <= max_n)] or all_idx
        target = gstep + per_stage
        print(f"--- stage {stage_i}: workers<={max_n or 'all'}  "
              f"({len(idx_pool)} clusters) ---")
        obs = env.reset(idx_pool[ep % len(idx_pool)])
        it = 0
        while gstep < target:
            it += 1
            critic_only = (stage_i == 2 and it <= args.critic_warmup)
            B = dict(o=[], m=[], a=[], lp=[], v=[], r=[], d=[])
            for _ in range(args.rollout):
                mask = env.active_mask
                o = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                m = torch.as_tensor(mask[None].astype(np.float32), device=device)
                policy.eval()
                _, raw_t, lp, val = policy.act(o, m)
                raw = raw_t[0].cpu().numpy()
                nobs, r, done, info = env.step(raw)
                B["o"].append(obs); B["m"].append(mask.astype(np.float32)); B["a"].append(raw)
                B["lp"].append(float(lp.item())); B["v"].append(float(val.item()))
                B["r"].append(r); B["d"].append(float(done))
                gstep += 1
                if done:
                    ep += 1
                    obs = env.reset(idx_pool[ep % len(idx_pool)])
                else:
                    obs = nobs

            with torch.no_grad():
                o = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                m = torch.as_tensor(env.active_mask[None].astype(np.float32), device=device)
                _, _, _, lastv = policy.act(o, m)
                lastv = float(lastv.item())

            rew = np.array(B["r"], np.float32); val = np.array(B["v"], np.float32)
            dn = np.array(B["d"], np.float32); adv = np.zeros_like(rew); g = 0.0
            for t in reversed(range(len(rew))):
                nv = lastv if t == len(rew) - 1 else val[t + 1]
                nt = 1.0 - dn[t]
                delta = rew[t] + args.gamma * nv * nt - val[t]
                g = delta + args.gamma * args.lam * nt * g
                adv[t] = g
            ret = adv + val
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            to = torch.as_tensor(np.array(B["o"]), dtype=torch.float32, device=device)
            tm = torch.as_tensor(np.array(B["m"]), dtype=torch.float32, device=device)
            ta = torch.as_tensor(np.array(B["a"]), dtype=torch.float32, device=device)
            tlp = torch.as_tensor(np.array(B["lp"]), dtype=torch.float32, device=device)
            tadv = torch.as_tensor(adv, device=device)
            tret = torch.as_tensor(ret, device=device)

            policy.train()
            n = len(rew); order = np.arange(n)
            for _ in range(args.epochs):
                np.random.shuffle(order)
                for st in range(0, n, args.minibatch):
                    b = order[st:st + args.minibatch]
                    lp, ent, vp = policy.evaluate_actions(to[b], tm[b], ta[b])
                    v_loss = ((vp - tret[b]) ** 2).mean()
                    if critic_only:
                        loss = v_loss
                    else:
                        ratio = torch.exp(lp - tlp[b])
                        s1 = ratio * tadv[b]
                        s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * tadv[b]
                        loss = (-torch.min(s1, s2).mean() + args.vf_coef * v_loss
                                - args.ent_coef * ent.mean())
                        if args.bc_coef > 0.0:
                            mean_now, _, _ = policy.raw_forward(to[b], tm[b])
                            with torch.no_grad():
                                mean_ref, _, _ = anchor.raw_forward(to[b], tm[b])
                            dec_now = decode_action(mean_now, cfg.max_forward_dist,
                                                    cfg.max_lateral_offset)
                            dec_ref = decode_action(mean_ref, cfg.max_forward_dist,
                                                    cfg.max_lateral_offset)
                            w = tm[b].unsqueeze(-1)
                            anchor_loss = ((((dec_now - dec_ref) / act_scale) ** 2)
                                           * w).sum() / w.sum().clamp(min=1.0)
                            loss = loss + args.bc_coef * anchor_loss
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    opt.step()

            if it % max(args.eval_every, 1) and gstep < target:
                continue
            ev = evaluate(env, policy, eval_idx, device, cfg=cfg)
            history.append(dict(step=gstep, stage=stage_i, **ev))
            print(f"  step {gstep:>7d} ret={ev['ret']:7.1f} resolved={ev['resolved']*100:5.1f}% "
                  f"coll={ev['collisions']:.2f} shield={ev['shield']*100:4.0f}% "
                  f"stop={ev['stop']*100:4.0f}%   (V0 ret={base['ret']:.1f})")
            if writer is not None:
                for k, v in ev.items():
                    writer.add_scalar(f"eval/{k}", v, gstep)
                writer.add_scalar("ref/v0_return", base["ret"], gstep)
            if ev["ret"] > best:
                best = ev["ret"]
                torch.save(dict(policy=policy.state_dict(), config=vars(cfg)),
                           _LOG / f"{args.name}_best.pt")

    torch.save(dict(policy=policy.state_dict(), config=vars(cfg)), _LOG / f"{args.name}_final.pt")
    (_LOG / f"{args.name}_history.json").write_text(json.dumps(
        dict(v0=base, bc=after_bc, history=history), indent=2, default=float))
    print(f"\nsaved -> {_LOG / (args.name + '_final.pt')}   best return {best:.1f} "
          f"(V0 teacher {base['ret']:.1f})")


if __name__ == "__main__":
    main()
