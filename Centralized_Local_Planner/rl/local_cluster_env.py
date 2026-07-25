"""Multi-robot RL env for spatial local PATH replanning (Phase 2).

Wraps the working `LocalReplanSim` (cluster -> busy-area lock -> 2D motion ->
rejoin). The learned attention policy controls the AMRs that are currently in
local-replan mode: it proposes a per-AMR local goal (desired displacement) that
the SAME 2D shield projects to a safe move. Cluster size varies frame to frame,
so observations are padded to ``num_amrs`` slots with an active mask (the
local-mode AMRs); attention + masking handle the variable agent count.

Workers are precomputed per seed (reusing the fleet-env cache) so RL stepping is
fast. The policy only shapes efficiency (clear the busy area quickly so waiting
AMRs are released sooner); safety is guaranteed by the shield.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tools.local_replanning import LocalReplanSim, LocalReplanConfig, ExternalReplanner
from ..tools.factory_map import MAP_BOUNDS
from .fleet_env import precompute_worker_data

OBS_DIM = 12


@dataclass
class LocalEnvConfig:
    num_frames: int = 420
    num_workers: int = 2
    num_amrs: int = 6
    max_step: float = 0.10           # m per frame (= max_speed*dt)
    w_prog: float = 6.0              # reward per metre closed toward exit
    w_rejoin: float = 5.0            # bonus per AMR that rejoins the rail
    w_time: float = 0.01             # penalty per local AMR per frame
    w_clear: float = 0.05            # clearance shaping
    w_collide: float = 5.0           # penalty per new collision (shield should prevent)


class LocalClusterEnv:
    def __init__(self, cfg: LocalEnvConfig | None = None):
        self.cfg = cfg or LocalEnvConfig()
        self.N = self.cfg.num_amrs
        self._diag = float(np.hypot(MAP_BOUNDS[1] - MAP_BOUNDS[0],
                                    MAP_BOUNDS[3] - MAP_BOUNDS[2]))

    def reset(self, seed: int):
        c = self.cfg
        frames, _, _, _ = precompute_worker_data(c.num_frames, c.num_workers, seed)
        self.replanner = ExternalReplanner(LocalReplanConfig(max_speed=c.max_step / 0.2))
        self.sim = LocalReplanSim(num_frames=c.num_frames, num_workers=c.num_workers,
                                  num_amrs=c.num_amrs, seed=seed,
                                  replanner=self.replanner, worker_frames=frames)
        self.frame = 0
        self._prev_dist = {}
        self._prev_collided = 0
        return self._observe()

    # -- observation ------------------------------------------------------
    def _observe(self):
        sim = self.sim; f = min(self.frame, self.cfg.num_frames - 1)
        obs = np.zeros((self.N, OBS_DIM), dtype=np.float32)
        mask = np.zeros(self.N, dtype=bool)
        local = [a for a in sim.amrs if a.local_mode and not a.collided]
        local_xy = {a.name: a.current_xy() for a in local}
        wpos = [(w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]) for w in sim.workers]
        for i, a in enumerate(sim.amrs):
            if not (a.local_mode and not a.collided):
                continue
            mask[i] = True
            xy = a.current_xy()
            goal = a.position_at(a.exit_s)
            to_exit = goal - xy
            de = float(np.linalg.norm(to_exit))
            # nearest worker
            nw = min(wpos, key=lambda wp: float(np.linalg.norm(xy - wp))) if wpos else xy
            rw = nw - xy; dw = float(np.linalg.norm(rw))
            # nearest peer
            peers = [local_xy[a2.name] for a2 in local if a2.name != a.name]
            if peers:
                npr = min(peers, key=lambda q: float(np.linalg.norm(xy - q)))
                rp = npr - xy; dp = float(np.linalg.norm(rp))
            else:
                rp = np.zeros(2); dp = self._diag
            obs[i] = [
                to_exit[0] / 5.0, to_exit[1] / 5.0, min(de / 5.0, 1.0),
                rw[0] / 3.0, rw[1] / 3.0, min(dw / self._diag, 1.0),
                rp[0] / 3.0, rp[1] / 3.0, min(dp / self._diag, 1.0),
                a.progress / max(a.total_length, 1e-9),
                (xy[0] - MAP_BOUNDS[0]) / (MAP_BOUNDS[1] - MAP_BOUNDS[0]),
                (xy[1] - MAP_BOUNDS[2]) / (MAP_BOUNDS[3] - MAP_BOUNDS[2]),
            ]
        self._mask = mask
        return obs

    @property
    def active_mask(self):
        return self._mask.copy()

    # -- step -------------------------------------------------------------
    def step(self, action: np.ndarray):
        c = self.cfg; sim = self.sim
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        # Direct control: the policy fully decides each in-cluster AMR's local
        # goal (displacement this step). No greedy prior / no residual -- RL
        # takes over path replanning once the cluster/busy-area is formed. The
        # 2D safety shield projects the move, so the policy can be aggressive.
        acts = {}
        dist_before = {}
        for i, a in enumerate(sim.amrs):
            if self._mask[i]:
                acts[a.name] = action[i] * c.max_step      # direct local goal
                dist_before[a.name] = float(np.linalg.norm(
                    a.current_xy() - a.position_at(a.exit_s)))
        self.replanner.set_actions(acts)

        names_local_before = set(dist_before.keys())
        sim.step(self.frame)

        # ---- reward ----
        prog = 0.0; rejoined = 0; n_local = 0; clear = 0.0
        f = min(self.frame, c.num_frames - 1)
        wpos = [(w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]) for w in sim.workers]
        for a in sim.amrs:
            if a.name in names_local_before:
                if not a.local_mode and not a.collided:
                    rejoined += 1                            # reached exit this frame
                elif a.local_mode:
                    n_local += 1
                    d_now = float(np.linalg.norm(a.current_xy() - a.position_at(a.exit_s)))
                    prog += dist_before[a.name] - d_now
                    if wpos:
                        dmin = min(float(np.linalg.norm(a.current_xy() - wp)) for wp in wpos)
                        clear += min(dmin, 1.5) / 1.5
        collided = sum(1 for a in sim.amrs if a.collided)
        new_coll = max(0, collided - self._prev_collided)
        self._prev_collided = collided

        reward = (c.w_prog * prog + c.w_rejoin * rejoined
                  - c.w_time * n_local + c.w_clear * (clear / max(n_local, 1))
                  - c.w_collide * new_coll)

        self.frame += 1
        done = (self.frame >= c.num_frames) or all(
            (a.is_done() or a.collided) for a in sim.amrs)
        obs = self._observe() if not done else np.zeros((self.N, OBS_DIM), dtype=np.float32)
        info = dict(
            completion=float(np.mean([a.is_done() for a in sim.amrs])),
            collided=collided,
            mean_progress=float(np.mean([a.progress / a.total_length for a in sim.amrs])),
        )
        return obs, float(reward), done, info
