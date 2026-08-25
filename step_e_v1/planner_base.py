"""Shared runtime for the V0 and V1 local replanners.

Both planners follow the same contract:

    1. receive a conflict cluster (variable N)
    2. propose one action per AMR       <-- the ONLY difference between V0/V1
    3. roll every action out into a short-horizon trajectory
    4. build backup candidates around it
    5. shield the whole cluster sequentially in priority order
    6. dispatch a command per AMR
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .commands import trajectory_to_command
from .safety_shield import select_safe_joint_trajectories
from .trajectory_rollout import generate_backup_candidates, rollout_action


class _LazyCandidates:
    """Proposal first; the backup set is materialised only when asked for."""

    __slots__ = ("first", "_build", "_full")

    def __init__(self, first, build):
        self.first = first
        self._build = build
        self._full = None

    def __call__(self):
        if self._full is None:
            self._full = self._build()
        return self._full

    @property
    def rollouts(self) -> int:
        """How many candidate rollouts were actually computed."""
        return len(self._full) if self._full is not None else 1

    def __len__(self):
        return len(self._full) if self._full is not None else 1

    def __getitem__(self, i):
        if i == 0 and self._full is None:
            return self.first
        return self()[i]

    def __iter__(self):
        yield self.first
        for item in self()[1:]:
            yield item


@dataclass
class PlanStats:
    plan_time_ms: float = 0.0
    n_agents: int = 0
    n_shield_modified: int = 0
    n_stop: int = 0
    n_unsafe: int = 0
    n_candidates: int = 0
    min_worker_clearance: float = float("inf")
    min_amr_distance: float = float("inf")
    proposals: dict = field(default_factory=dict)      # id -> proposed action
    executed: dict = field(default_factory=dict)       # id -> executed action


class ShieldedReplanner:
    """Base class: rollout + backups + joint shield + command dispatch."""

    name = "base"
    # V0 ranks every safe candidate by cost; V1 executes its own proposal when
    # it is safe (see select_safe_trajectory).
    prefer_proposal = False

    def __init__(self, config):
        self.config = config
        self.stats = PlanStats()
        self.last_action: dict[str, np.ndarray] = {}

    # -- to be provided by subclasses --------------------------------------
    def propose_actions(self, cluster, worker_predictions, map_data) -> dict:
        raise NotImplementedError

    def priority_score(self, agent) -> float:
        """Higher = plans (and reserves space) first."""
        c = self.config
        ttc = agent.ttc if np.isfinite(agent.ttc) else 1e6
        return (c.prio_alpha / (ttc + c.ttc_eps)
                + c.prio_beta * float(agent.task_priority)
                + c.prio_gamma * float(agent.braking_risk))

    # -- runtime ------------------------------------------------------------
    def plan(self, cluster, worker_predictions, map_data, dt=None) -> dict:
        """Returns ``{amr_id: Command}`` for every AMR in the cluster."""
        cfg = self.config
        dt = cfg.dt if dt is None else float(dt)
        t0 = time.perf_counter()
        cluster = list(cluster)
        if not cluster:
            self.stats = PlanStats()
            return {}

        proposals = self.propose_actions(cluster, worker_predictions, map_data)

        lazy = self.prefer_proposal and cfg.lazy_backups
        candidates = {}
        for agent in cluster:
            a0 = np.asarray(proposals[agent.id], dtype=float)

            def build(agent=agent, a0=a0):
                acts = (generate_backup_candidates(a0, cfg)
                        if cfg.use_backup_candidates else [a0, np.zeros(3)])
                return [(a, rollout_action(agent, a, cfg.horizon_sec, dt, cfg, map_data))
                        for a in acts]

            if lazy:
                # Roll out the proposal now; the shield expands the backups
                # only if the proposal turns out to be unsafe.
                first = (a0, rollout_action(agent, a0, cfg.horizon_sec, dt, cfg, map_data))

                def lazy_list(first=first, build=build):
                    rest = build()
                    return [first] + rest[1:]

                candidates[agent.id] = _LazyCandidates(first, lazy_list)
            else:
                candidates[agent.id] = build()

        order = [a.id for a in sorted(cluster, key=self.priority_score, reverse=True)]
        results = select_safe_joint_trajectories(
            cluster, candidates, worker_predictions, map_data, cfg, order=order,
            prefer_proposal=self.prefer_proposal, prev_actions=self.last_action)

        commands, stats = {}, PlanStats(n_agents=len(cluster))
        for agent in cluster:
            res = results[agent.id]
            cmd = trajectory_to_command(agent, res.trajectory, res, cfg)
            commands[agent.id] = cmd
            stats.n_shield_modified += int(res.modified)
            stats.n_unsafe += int(not res.safe)
            stats.n_stop += int(cmd.mode == "STOP")
            cand = candidates[agent.id]
            stats.n_candidates += (cand.rollouts if isinstance(cand, _LazyCandidates)
                                   else len(cand))
            stats.min_worker_clearance = min(stats.min_worker_clearance,
                                             res.report.min_worker_clearance)
            stats.min_amr_distance = min(stats.min_amr_distance,
                                         res.report.min_amr_distance)
            stats.proposals[agent.id] = np.asarray(proposals[agent.id], float)
            stats.executed[agent.id] = np.asarray(res.action, float)
            self.last_action[agent.id] = np.asarray(res.action, float)
        stats.plan_time_ms = (time.perf_counter() - t0) * 1000.0
        self.stats = stats
        return commands
