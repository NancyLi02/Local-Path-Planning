"""Multi-robot RL env for spatial local PATH replanning (Phase 2).

Wraps the working `LocalReplanSim` (cluster -> busy-area lock -> 2D motion ->
rejoin). The learned attention policy controls the AMRs that are currently in
local-replan mode: it proposes a per-AMR local goal (desired displacement) that
the SAME 2D shield projects to a safe move.

Observations preserve CLUSTER MEMBERSHIP: one row per busy-area lock, padded to
``num_amrs`` member slots. A cluster only ever sees its own members -- rows are
independent batch entries for the attention pass (see ``rl/cluster_policy.py``)
and the nearest-peer features are computed within the row. Cluster count and
cluster size both vary frame to frame, hence ``cluster_valid`` (which rows hold
a cluster) alongside ``cluster_mask`` (which slots hold an AMR).

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
ACT_DIM = 2          # mirrors rl.local_policy.ACT_DIM; kept here so this env stays torch-free


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
    """obs (C, M, OBS_DIM); act (C, M, ACT_DIM) local goals, one row per cluster."""

    def __init__(self, cfg: LocalEnvConfig | None = None):
        self.cfg = cfg or LocalEnvConfig()
        self.N = self.cfg.num_amrs                       # member slots per cluster
        # Locks are disjoint and form with >= 2 members, so this bounds the
        # number of simultaneous clusters.
        self.C = max(1, self.cfg.num_amrs // 2)
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

    # -- cluster bookkeeping ----------------------------------------------
    def _cluster_groups(self) -> list[list]:
        """Live local-mode members of each busy area, one list per cluster.

        ``sim.locks`` is the authoritative cluster identity: membership is
        frozen when the lock forms, whereas Step D rebuilds its clusters from
        scratch every frame.
        """
        sim = self.sim
        by_name = {a.name: a for a in sim.amrs}
        groups: list[list] = []
        claimed: set[str] = set()
        for lk in sim.locks:
            members = [by_name[n] for n in sorted(lk.members) if n in by_name]
            members = [a for a in members if a.local_mode and not a.collided]
            if members:
                groups.append(members)
                claimed.update(a.name for a in members)
        # An AMR only enters local mode together with a lock, so this should not
        # trigger; give any stray its own cluster rather than letting it borrow
        # another cluster's context.
        for a in sim.amrs:
            if a.local_mode and not a.collided and a.name not in claimed:
                groups.append([a])
        return groups

    # -- observation ------------------------------------------------------
    def _member_features(self, amr, members: list, wpos: list) -> list[float]:
        """Feature vector for ``amr``; peers are restricted to its own cluster."""
        xy = amr.current_xy()
        goal = amr.position_at(amr.exit_s)
        to_exit = goal - xy
        de = float(np.linalg.norm(to_exit))
        # nearest worker
        nw = min(wpos, key=lambda wp: float(np.linalg.norm(xy - wp))) if wpos else xy
        rw = nw - xy; dw = float(np.linalg.norm(rw))
        # nearest peer INSIDE this cluster
        peers = [p.current_xy() for p in members if p.name != amr.name]
        if peers:
            npr = min(peers, key=lambda q: float(np.linalg.norm(xy - q)))
            rp = npr - xy; dp = float(np.linalg.norm(rp))
        else:
            rp = np.zeros(2); dp = self._diag
        return [
            to_exit[0] / 5.0, to_exit[1] / 5.0, min(de / 5.0, 1.0),
            rw[0] / 3.0, rw[1] / 3.0, min(dw / self._diag, 1.0),
            rp[0] / 3.0, rp[1] / 3.0, min(dp / self._diag, 1.0),
            amr.progress / max(amr.total_length, 1e-9),
            (xy[0] - MAP_BOUNDS[0]) / (MAP_BOUNDS[1] - MAP_BOUNDS[0]),
            (xy[1] - MAP_BOUNDS[2]) / (MAP_BOUNDS[3] - MAP_BOUNDS[2]),
        ]

    def _blank(self):
        self._cobs = np.zeros((self.C, self.N, OBS_DIM), dtype=np.float32)
        self._cmask = np.zeros((self.C, self.N), dtype=bool)
        self._cvalid = np.zeros(self.C, dtype=bool)
        self._cnames: list[list[str]] = [[] for _ in range(self.C)]
        return self._cobs

    def _observe(self):
        sim = self.sim; f = min(self.frame, self.cfg.num_frames - 1)
        wpos = [(w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]) for w in sim.workers]
        obs = np.zeros((self.C, self.N, OBS_DIM), dtype=np.float32)
        cmask = np.zeros((self.C, self.N), dtype=bool)
        cvalid = np.zeros(self.C, dtype=bool)
        cnames: list[list[str]] = [[] for _ in range(self.C)]
        for c, members in enumerate(self._cluster_groups()[:self.C]):
            cvalid[c] = True
            for m, amr in enumerate(members[:self.N]):
                cmask[c, m] = True
                cnames[c].append(amr.name)
                obs[c, m] = self._member_features(amr, members, wpos)
        self._cobs = obs; self._cmask = cmask
        self._cvalid = cvalid; self._cnames = cnames
        return obs

    @property
    def cluster_obs(self):
        return self._cobs.copy()

    @property
    def cluster_mask(self):
        """(C, M) bool -- which member slots hold a real AMR."""
        return self._cmask.copy()

    @property
    def cluster_valid(self):
        """(C,) bool -- which rows hold a cluster this frame."""
        return self._cvalid.copy()

    @property
    def cluster_names(self):
        """cluster_names[c][m] = name of the AMR in slot m of cluster c."""
        return [list(n) for n in self._cnames]

    @property
    def n_clusters(self) -> int:
        return int(self._cvalid.sum())

    # -- step -------------------------------------------------------------
    def step(self, action: np.ndarray):
        cfg = self.cfg; sim = self.sim
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        action = action.reshape(self.C, self.N, ACT_DIM)
        by_name = {a.name: a for a in sim.amrs}

        # Direct control: the policy fully decides each in-cluster AMR's local
        # goal (displacement this step). No greedy prior / no residual -- RL
        # takes over path replanning once the cluster/busy-area is formed. The
        # 2D safety shield projects the move, so the policy can be aggressive.
        # Actions are dispatched by name, so a cluster's row never leaks into
        # another cluster's members.
        acts = {}
        dist_before = {}
        for c in range(self.C):
            if not self._cvalid[c]:
                continue
            for m, name in enumerate(self._cnames[c]):
                amr = by_name[name]
                acts[name] = action[c, m] * cfg.max_step      # direct local goal
                dist_before[name] = float(np.linalg.norm(
                    amr.current_xy() - amr.position_at(amr.exit_s)))
        self.replanner.set_actions(acts)

        names_local_before = set(dist_before.keys())
        sim.step(self.frame)

        # ---- reward ----
        prog = 0.0; rejoined = 0; n_local = 0; clear = 0.0
        f = min(self.frame, cfg.num_frames - 1)
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

        reward = (cfg.w_prog * prog + cfg.w_rejoin * rejoined
                  - cfg.w_time * n_local + cfg.w_clear * (clear / max(n_local, 1))
                  - cfg.w_collide * new_coll)

        self.frame += 1
        done = (self.frame >= cfg.num_frames) or all(
            (a.is_done() or a.collided) for a in sim.amrs)
        obs = self._observe() if not done else self._blank()
        info = dict(
            completion=float(np.mean([a.is_done() for a in sim.amrs])),
            collided=collided,
            mean_progress=float(np.mean([a.progress / a.total_length for a in sim.amrs])),
        )
        return obs, float(reward), done, info
