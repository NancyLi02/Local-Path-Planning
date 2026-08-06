"""Per-cluster EPISODIC env for the spatial local-replanning policy.

The full-sim env gives very sparse 'cluster-resolution' signal (clusters are
rare; one stuck AMR locks everyone forever), so from-scratch RL stalls. Here
each EPISODE is one harvested cluster: the member AMRs start at their entry
positions and must reach their downstream rejoin points (exits) while workers
move through, behind the same 2D shield. Dense rejoin signal + short episodes
=> learnable from scratch.

Observation layout is IDENTICAL to LocalClusterEnv (12-dim) so the trained
policy transfers directly to the full simulator for deployment / video.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..tools.local_replanning import (
    LocalReplanSim, LocalReplanConfig, shielded_displacement)
from ..tools.scenario import make_workers
from ..tools.factory_map import MAP_BOUNDS

OBS_DIM = 12
MAX_AGENTS = 6
_CACHE = Path(__file__).resolve().parents[2] / ".cache"


@dataclass
class EpisodicConfig:
    horizon: int = 90               # max frames to resolve a cluster
    max_step: float = 0.10
    exit_tol: float = 0.60
    worker_collision_dist: float = 0.55
    w_prog: float = 6.0
    w_rejoin: float = 5.0
    w_time: float = 0.01
    w_clear: float = 0.05
    w_collide: float = 5.0


# ---------------------------------------------------------------------------
# Harvest cluster scenarios from the rule simulator (once, cached)
# ---------------------------------------------------------------------------

def harvest_scenarios(seeds, num_frames=420, num_workers=2, num_amrs=6):
    tag = f"clusters_s{min(seeds)}-{max(seeds)}_f{num_frames}_w{num_workers}_a{num_amrs}"
    cache = _CACHE / f"{tag}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    scen = []
    for s in seeds:
        sim = LocalReplanSim(num_frames=num_frames, num_workers=num_workers,
                             num_amrs=num_amrs, seed=s)
        prog_at = {}
        for f in range(num_frames):
            # record progress fraction just before stepping (entry snapshot)
            for a in sim.amrs:
                prog_at[a.name] = a.progress / max(a.total_length, 1e-9)
            sim.step(f)
            for nf in getattr(sim, "newly_formed", []):
                members = [dict(name=m["name"], color=m["color"],
                                entry=np.asarray(m["entry"], float),
                                exit=np.asarray(m["exit"], float),
                                prog=float(prog_at.get(m["name"], 0.0)))
                           for m in nf["members"]]
                scen.append(dict(seed=s, frame=nf["frame"], members=members))
    _CACHE.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(scen))
    return scen


# ---------------------------------------------------------------------------
# Episodic environment
# ---------------------------------------------------------------------------

class EpisodicLocalEnv:
    def __init__(self, cfg: EpisodicConfig | None = None, seeds=range(8),
                 num_frames=420, num_workers=2, num_amrs=6):
        self.cfg = cfg or EpisodicConfig()
        self.scen = harvest_scenarios(list(seeds), num_frames, num_workers, num_amrs)
        self._worker_truth = {}      # seed -> list of truth arrays
        self._nw = num_workers
        self._nf = num_frames
        self._diag = float(np.hypot(MAP_BOUNDS[1] - MAP_BOUNDS[0],
                                    MAP_BOUNDS[3] - MAP_BOUNDS[2]))
        self._rcfg = LocalReplanConfig(max_speed=self.cfg.max_step / 0.2,
                                       exit_tol=self.cfg.exit_tol)
        self.N = MAX_AGENTS

    def _workers(self, seed):
        if seed not in self._worker_truth:
            ws = make_workers(self._nf, 0.2, self._nw)
            # make_workers ignores seed for layout but truth is deterministic;
            # regenerate per seed for the matching scenario worker motion.
            self._worker_truth[seed] = [w["truth"] for w in ws]
        return self._worker_truth[seed]

    def n_scenarios(self):
        return len(self.scen)

    def reset(self, sel: int):
        sc = self.scen[sel % len(self.scen)]
        self.seed = sc["seed"]; self.t0 = sc["frame"]
        self.truth = self._workers(self.seed)
        self.k = 0
        self.agents = []
        for m in sc["members"][:self.N]:
            self.agents.append(dict(xy=m["entry"].copy(), exit=m["exit"].copy(),
                                    prog=m["prog"], done=False, collided=False))
        return self._observe()

    def _worker_pos(self):
        f = min(self.t0 + self.k, self._nf - 1)
        return [t[f] if f < len(t) else t[-1] for t in self.truth]

    def _observe(self):
        wpos = self._worker_pos()
        obs = np.zeros((self.N, OBS_DIM), dtype=np.float32)
        mask = np.zeros(self.N, dtype=bool)
        live = [g for g in self.agents if not g["done"] and not g["collided"]]
        for i, g in enumerate(self.agents):
            if g["done"] or g["collided"]:
                continue
            mask[i] = True
            xy = g["xy"]; to_exit = g["exit"] - xy; de = float(np.linalg.norm(to_exit))
            nw = min(wpos, key=lambda wp: float(np.linalg.norm(xy - wp)))
            rw = nw - xy; dw = float(np.linalg.norm(rw))
            peers = [o["xy"] for o in live if o is not g]
            if peers:
                npr = min(peers, key=lambda q: float(np.linalg.norm(xy - q)))
                rp = npr - xy; dp = float(np.linalg.norm(rp))
            else:
                rp = np.zeros(2); dp = self._diag
            obs[i] = [to_exit[0] / 5.0, to_exit[1] / 5.0, min(de / 5.0, 1.0),
                      rw[0] / 3.0, rw[1] / 3.0, min(dw / self._diag, 1.0),
                      rp[0] / 3.0, rp[1] / 3.0, min(dp / self._diag, 1.0),
                      g["prog"],
                      (xy[0] - MAP_BOUNDS[0]) / (MAP_BOUNDS[1] - MAP_BOUNDS[0]),
                      (xy[1] - MAP_BOUNDS[2]) / (MAP_BOUNDS[3] - MAP_BOUNDS[2])]
        self._mask = mask
        return obs

    @property
    def active_mask(self):
        return self._mask.copy()

    def step(self, action):
        c = self.cfg
        action = np.clip(np.asarray(action, float), -1.0, 1.0)
        wpos = self._worker_pos()
        live_idx = [i for i in range(self.N) if self._mask[i]]
        # priority: nearest to exit first (shapes reservation order)
        live_idx.sort(key=lambda i: float(np.linalg.norm(
            self.agents[i]["xy"] - self.agents[i]["exit"])))
        reserved = []
        prog = 0.0; rejoined = 0; collided_new = 0; clear = 0.0; n_active = len(live_idx)
        for i in live_idx:
            g = self.agents[i]
            d_before = float(np.linalg.norm(g["xy"] - g["exit"]))
            desired = action[i] * c.max_step
            stepn = min(c.max_step, float(np.linalg.norm(desired)))
            new_xy, _ = shielded_displacement(g["xy"], desired, stepn, wpos, reserved, self._rcfg)
            g["xy"] = new_xy; reserved.append(new_xy)
            # collision (shield should prevent; keep honest)
            dmin = min(float(np.linalg.norm(new_xy - wp)) for wp in wpos)
            if dmin < c.worker_collision_dist:
                g["collided"] = True; collided_new += 1; continue
            clear += min(dmin, 1.5) / 1.5
            d_now = float(np.linalg.norm(g["xy"] - g["exit"]))
            prog += d_before - d_now
            if d_now < c.exit_tol:
                g["done"] = True; rejoined += 1

        reward = (c.w_prog * prog + c.w_rejoin * rejoined
                  - c.w_time * n_active + c.w_clear * (clear / max(n_active, 1))
                  - c.w_collide * collided_new)
        self.k += 1
        done = self.k >= c.horizon or all(g["done"] or g["collided"] for g in self.agents)
        obs = self._observe() if not done else np.zeros((self.N, OBS_DIM), dtype=np.float32)
        n_total = len(self.agents)
        info = dict(completion=sum(g["done"] for g in self.agents) / max(n_total, 1),
                    collided=sum(g["collided"] for g in self.agents),
                    mean_progress=sum(g["done"] for g in self.agents) / max(n_total, 1))
        return obs, float(reward), done, info
