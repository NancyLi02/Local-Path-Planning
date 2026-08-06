"""Render an episode of the multi-robot cluster env to an mp4 (eval videos)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from .env import MultiRobotClusterEnv, ClusterEnvConfig

_PALETTE = ["#0aa6a6", "#2d3e8c", "#7b1fa2", "#c2185b", "#5d6d00", "#00897b"]


def rollout_snapshots(env, policy, seed, device):
    import torch
    obs = env.reset(seed)
    rails = [(r.a.copy(), r.b.copy()) for r in env.rails]
    snaps = []
    done = False; info = {}
    while not done:
        mask = env.active_mask
        if policy is None:
            act = np.zeros((env.N, 2), dtype=np.float32)   # raw 0 -> mid fwd, no lat
        else:
            ot = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
            mt = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
            a, _, _ = policy.act(ot, mt, deterministic=True)
            act = a[0].cpu().numpy()
        obs, r, done, info = env.step(act)
        snaps.append(dict(
            amr=[dict(x=a["x"], y=a["y"], th=a["th"], done=a["done"], col=a["collided"])
                 for a in env.amr],
            hum=[(h["x"], h["y"]) for h in env.humans],
            step=env.steps))
    return rails, snaps, info


def render_episode(env, policy, seed, out_path, device="cpu"):
    rails, snaps, info = rollout_snapshots(env, policy, seed, device)
    n = len(rails); ms = env.cfg.map_size
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=110)
    ax.set_xlim(ms / 2 - 8, ms / 2 + 8); ax.set_ylim(ms / 2 - 8, ms / 2 + 8)
    ax.set_aspect("equal"); ax.set_facecolor("#f7fbff")
    ax.set_title("Multi-robot cluster local replanning (RL)")
    for i, (a, b) in enumerate(rails):
        col = _PALETTE[i % len(_PALETTE)]
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color=col, lw=1.2, alpha=0.5)
        ax.scatter(*b, marker="o", s=45, facecolor=col, edgecolor="white", zorder=2)

    bodies, halos, trails = [], [], []
    for i in range(n):
        col = _PALETTE[i % len(_PALETTE)]
        b = Circle((0, 0), env.cfg.robot_radius, facecolor=col, edgecolor="white", lw=1.3, zorder=6)
        h = Circle((0, 0), 0.55, facecolor="none", edgecolor=col, lw=1.5, alpha=0.4, zorder=5)
        t, = ax.plot([], [], "-", color=col, lw=1.4, alpha=0.7, zorder=3)
        ax.add_patch(b); ax.add_patch(h); bodies.append(b); halos.append(h); trails.append(t)
    arrows = [ax.plot([], [], "-", color="k", lw=1.2, zorder=7)[0] for _ in range(n)]
    nh = max(len(s["hum"]) for s in snaps)
    humans = [Circle((0, 0), env.cfg.human_radius, facecolor="#d84315", edgecolor="white", lw=1.0, zorder=8)
              for _ in range(nh)]
    for hh in humans:
        ax.add_patch(hh)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9,
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="#ccc"))
    paths = [[] for _ in range(n)]

    res = "SUCCESS" if info.get("success") else ("COLLISION" if info.get("collision") else "timeout")

    def update(f):
        s = snaps[f]
        for i, a in enumerate(s["amr"]):
            paths[i].append((a["x"], a["y"]))
            bodies[i].center = (a["x"], a["y"])
            halos[i].center = (a["x"], a["y"])
            if a["col"]:
                bodies[i].set_facecolor("#212121")
            tr = np.array(paths[i]); trails[i].set_data(tr[:, 0], tr[:, 1])
            arrows[i].set_data([a["x"], a["x"] + 0.5 * math.cos(a["th"])],
                               [a["y"], a["y"] + 0.5 * math.sin(a["th"])])
        for hi in range(nh):
            if hi < len(s["hum"]):
                humans[hi].center = s["hum"][hi]; humans[hi].set_alpha(1.0)
            else:
                humans[hi].set_alpha(0.0)
        hud.set_text(f"step {s['step']}  AMRs {n}  -> {res}")
        return bodies + halos + trails + arrows + humans + [hud]

    anim = FuncAnimation(fig, update, frames=len(snaps), interval=80, blit=False)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=15, bitrate=1800)); saved = out_path
    except Exception:
        saved = out_path.with_suffix(".gif"); anim.save(saved, writer=PillowWriter(fps=12))
    plt.close(fig)
    return saved, res, info
