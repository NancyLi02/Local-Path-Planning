"""Cluster-level RL environment for training the attention policy.

One EPISODE is one conflict cluster: the environment restores a snapshot of
the full simulator taken at the moment a cluster formed, and the policy then
proposes the joint action every frame until the cluster resolves (all members
rejoined their reference paths, collided, or the horizon expired).

Why episodic: cluster events are rare inside a 420-frame run, so training on
whole runs gives a very sparse learning signal. Snapshots keep the dynamics,
the shield and the command tracking exactly identical to deployment while
concentrating the experience on the frames the policy actually controls.

The reward is cluster-level, as the specification requires:

    r = - collision - worker_risk - amr_risk - delay - route_deviation
        - jerk - shield_penalty + progress + reach

The shield penalty is essential: without it the policy learns to output unsafe
actions and let the shield clean up after it.
"""
from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.action_decoder import decode_action_np
from ..common.config import PlannerConfig
from ..common.observation_builder import build_cluster_observation_np
from ..common.planner_base import ShieldedReplanner
from ..common.runtime import LocalReplanningRuntime
from ..optimization_based.planner import OptimizationBasedReplanner
from ..common.worker_cache import worker_frames

_CACHE = Path(__file__).resolve().parents[3] / ".cache" / "local_path_replanning"


class ExternalProposalPlanner(ShieldedReplanner):
    """Replanner whose proposals are injected from outside (the RL policy)."""

    name = "external"
    prefer_proposal = True

    def __init__(self, config):
        super().__init__(config)
        self.pending: dict[str, np.ndarray] = {}
        self._fallback = OptimizationBasedReplanner(config)

    def set_actions(self, actions: dict) -> None:
        self.pending = dict(actions)

    def propose_actions(self, cluster, worker_predictions, map_data) -> dict:
        base = self._fallback.propose_actions(cluster, worker_predictions, map_data)
        for agent in cluster:
            if agent.id in self.pending:
                base[agent.id] = np.asarray(self.pending[agent.id], dtype=float)
        return base


@dataclass
class EnvConfig:
    frames: int = 420
    num_workers: int = 2
    num_amrs: int = 6
    episode_frames: int = 120        # cap on one cluster episode
    # Scene variants to harvest clusters from. The scripted workers are
    # deterministic, so changing only the seed barely changes the scene -- the
    # variety that matters comes from the worker COUNT and the fleet size.
    worker_counts: tuple = (1, 2, 3)
    amr_counts: tuple = (4, 6)


class ClusterEnv:
    """Episodic, variable-N cluster environment (padded to ``max_agents``)."""

    def __init__(self, cfg: PlannerConfig, env_cfg: EnvConfig | None = None,
                 seeds=range(4), use_disk: bool = True):
        self.cfg = cfg
        self.env_cfg = env_cfg or EnvConfig()
        self.seeds = list(seeds)
        self.snapshots = self._harvest(use_disk)
        self.N = cfg.max_agents

    # -- snapshot harvesting -------------------------------------------------
    def scenes(self) -> list[tuple[int, int, int]]:
        e = self.env_cfg
        return [(seed, w, a) for seed in self.seeds
                for w in e.worker_counts for a in e.amr_counts]

    def _harvest(self, use_disk: bool) -> list:
        e = self.env_cfg
        tag = (f"snap_s{min(self.seeds)}-{max(self.seeds)}_f{e.frames}"
               f"_w{'-'.join(map(str, e.worker_counts))}"
               f"_a{'-'.join(map(str, e.amr_counts))}")
        path = _CACHE / f"{tag}.pkl"
        if use_disk and path.exists():
            return pickle.loads(path.read_bytes())
        snaps = []
        for seed, n_w, n_a in self.scenes():
            wf = worker_frames(e.frames, n_w, seed)
            rt = LocalReplanningRuntime(OptimizationBasedReplanner(self.cfg), self.cfg,
                              num_frames=e.frames, num_workers=n_w,
                              num_amrs=n_a, seed=seed, worker_frames=wf)
            for f in range(e.frames):
                rt.step(f)
                for nf in rt.newly_formed:
                    snaps.append(dict(seed=seed, workers=n_w, amrs=n_a,
                                      frame=f + 1, cluster_id=nf["cluster_id"],
                                      members=list(nf["members"]),
                                      state=_snapshot(rt)))
        if use_disk:
            _CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(snaps))
        return snaps

    def n_scenarios(self) -> int:
        return len(self.snapshots)

    # -- gym-ish API ---------------------------------------------------------
    def reset(self, index: int):
        snap = self.snapshots[index % len(self.snapshots)]
        e = self.env_cfg
        n_w = snap.get("workers", e.num_workers)
        n_a = snap.get("amrs", e.num_amrs)
        wf = worker_frames(e.frames, n_w, snap["seed"])
        self.planner = ExternalProposalPlanner(self.cfg)
        self.rt = LocalReplanningRuntime(self.planner, self.cfg, num_frames=e.frames,
                               num_workers=n_w, num_amrs=n_a,
                               seed=snap["seed"], worker_frames=wf)
        _restore(self.rt, snap["state"])
        self.frame = snap["frame"]
        self.t0 = snap["frame"]
        self.members = list(snap["members"])
        self._collided0 = sum(a.collided for a in self.rt.amrs)
        self._done_before = self._n_released()
        return self._observe()

    def _n_released(self) -> int:
        return sum(1 for n in self.members if n not in self.rt.control)

    def _observe(self):
        agents = [a for a in self.rt.current_cluster_agents()
                  if a.id in self.members]
        self._agents = agents[: self.N]
        obs, mask, ids = build_cluster_observation_np(
            self._agents, self.rt.last_worker_predictions, self.rt.map_data, self.cfg)
        self._ids = ids
        self._mask = mask[0].astype(bool)
        return obs[0]

    @property
    def active_mask(self) -> np.ndarray:
        return self._mask.copy()

    def step(self, raw_action: np.ndarray):
        cfg = self.cfg
        # The policy works in RAW (pre-decode) space; the planner expects the
        # bounded action, so decode here -- exactly as LearningBasedReplanner does
        # online, which keeps training and deployment identical.
        acts = {}
        for k, amr_id in enumerate(self._ids):
            acts[amr_id] = decode_action_np(raw_action[k], cfg.max_forward_dist,
                                            cfg.max_lateral_offset)
        self.planner.set_actions(acts)

        s_before = {n: self.rt.control[n].s for n in self.members if n in self.rt.control}
        out = self.rt.step(self.frame)
        self.frame += 1
        log = out["log"]

        # ---- cluster-level reward -----------------------------------------
        n_ctrl = max(len(s_before), 1)
        progress = 0.0
        deviation = 0.0
        jerk = 0.0
        delay = 0.0
        for n in s_before:
            st = self.rt.control.get(n)
            if st is None:
                continue
            progress += max(st.s - s_before[n], 0.0)
            deviation += abs(st.lat) / max(cfg.max_lateral_offset, 1e-9)
            jerk += min(abs(st.accel) / max(cfg.a_max, 1e-9), 2.0)
            delay += 1.0 - min(st.speed / max(cfg.v_max, 1e-9), 1.0)

        worker_risk = amr_risk = shield = 0.0
        for n in s_before:
            cmd = log.commands.get(n)
            if cmd is None:
                continue
            worker_risk += float(np.clip(1.0 - cmd.min_worker_clearance / 1.5, 0.0, 1.0))
            if np.isfinite(cmd.min_amr_distance):
                amr_risk += float(np.clip(
                    1.0 - cmd.min_amr_distance / (cfg.min_amr_distance * 1.5), 0.0, 1.0))
            shield += float(cmd.shield_modified)

        collided = sum(a.collided for a in self.rt.amrs) - self._collided0
        self._collided0 += collided
        released = self._n_released() - self._done_before
        self._done_before += released

        reward = (cfg.r_progress * progress
                  + cfg.r_reach * released
                  - cfg.r_collision * collided
                  - cfg.r_worker_risk * worker_risk / n_ctrl
                  - cfg.r_amr_risk * amr_risk / n_ctrl
                  - cfg.r_delay * delay / n_ctrl
                  - cfg.r_route_deviation * deviation / n_ctrl
                  - cfg.r_jerk * jerk / n_ctrl
                  - cfg.r_shield * shield / n_ctrl)

        obs = self._observe()
        alive = any(n in self.rt.control for n in self.members)
        done = (not alive
                or self.frame - self.t0 >= self.env_cfg.episode_frames
                or self.frame >= self.env_cfg.frames - 1)
        info = dict(
            resolved=float(self._n_released()) / max(len(self.members), 1),
            collisions=int(sum(a.collided for a in self.rt.amrs
                               if a.name in self.members)),
            shield_rate=shield / n_ctrl,
            stop_rate=sum(1 for n in s_before
                          if (log.commands.get(n) is not None
                              and log.commands[n].mode == "STOP")) / n_ctrl,
            frames=self.frame - self.t0,
        )
        return obs, float(reward), bool(done), info


# ---------------------------------------------------------------------------
# Snapshot helpers (a cluster episode must restart from an exact sim state)
# ---------------------------------------------------------------------------

def _snapshot(rt: LocalReplanningRuntime) -> dict:
    return dict(amrs=copy.deepcopy(rt.amrs), control=copy.deepcopy(rt.control),
                locks=copy.deepcopy(rt.locks), next_id=rt._next_cluster_id,
                qr=copy.deepcopy(rt.qr_planner))


def _restore(rt: LocalReplanningRuntime, state: dict) -> None:
    rt.amrs = copy.deepcopy(state["amrs"])
    rt.control = copy.deepcopy(state["control"])
    rt.locks = copy.deepcopy(state["locks"])
    rt._next_cluster_id = state["next_id"]
    rt.qr_planner = copy.deepcopy(state["qr"])
