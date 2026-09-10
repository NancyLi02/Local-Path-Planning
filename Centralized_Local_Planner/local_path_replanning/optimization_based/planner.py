"""Optimization-based: TTC-priority sequential replanning over a candidate set.

    priority_i = alpha / (TTC_i + eps) + beta * task_priority_i
                 + gamma * braking_risk_i

AMRs are planned one at a time in descending priority. Each one proposes a
nominal action (from the optional single-agent policy, otherwise "drive to the
local goal at full speed"), expands it into the specification's candidate set,
and commits the lowest-cost candidate that passes the shield; the accepted
trajectory is then reserved so lower-priority AMRs treat it as a moving
obstacle. STOP is the final fallback.

Deterministic and interpretable -- the engineering baseline the learned planner must
beat on coordination quality while matching its safety.
"""
from __future__ import annotations

import numpy as np

from ..common.planner_base import ShieldedReplanner


class OptimizationBasedReplanner(ShieldedReplanner):
    name = "optimization_based"

    def __init__(self, config, single_policy=None):
        super().__init__(config)
        # Optional single-AMR policy a_i^0 = pi_single(o_i). When absent the
        # nominal proposal is the greedy "keep going to the local goal".
        self.single_policy = single_policy

    def propose_actions(self, cluster, worker_predictions, map_data) -> dict:
        cfg = self.config
        out = {}
        for agent in cluster:
            if self.single_policy is not None:
                out[agent.id] = np.asarray(
                    self.single_policy(agent, cluster, worker_predictions,
                                       map_data, cfg), dtype=float)
                continue
            fwd = float(np.clip(agent.goal_s - agent.s, 0.0, cfg.max_forward_dist))
            if fwd < 1e-3:
                fwd = cfg.max_forward_dist
            out[agent.id] = np.array([fwd, 0.0, 1.0])      # full speed, on-path
        return out
