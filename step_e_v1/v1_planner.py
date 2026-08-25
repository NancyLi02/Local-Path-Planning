"""V1: attention-based shared policy replanner.

The joint action proposal comes from ``MultiAMRAttentionPolicy``; everything
downstream (rollout, backup candidates, space-time shield, command dispatch)
is identical to V0, so any difference in the measured behaviour is caused by
the policy alone.

    V1 attention policy proposes joint actions.
    The safety shield filters executable trajectories.
"""
from __future__ import annotations

import numpy as np
import torch

from .attention_policy import MultiAMRAttentionPolicy
from .observation_builder import build_cluster_observation
from .planner_base import ShieldedReplanner


class V1AttentionReplanner(ShieldedReplanner):
    name = "V1"
    prefer_proposal = True

    def __init__(self, policy, config, device: str = "cpu", deterministic: bool = True):
        super().__init__(config)
        self.prefer_proposal = bool(getattr(config, "v1_prefer_proposal", True))
        self.policy = policy.to(device)
        self.device = device
        self.deterministic = bool(deterministic)
        self.policy.eval()
        self.last_value: float = 0.0

    @classmethod
    def from_checkpoint(cls, path, config, device: str = "cpu", **kw):
        policy = MultiAMRAttentionPolicy(config)
        state = torch.load(path, map_location=device)
        policy.load_state_dict(state["policy"] if "policy" in state else state)
        return cls(policy, config, device=device, **kw)

    def propose_actions(self, cluster, worker_predictions, map_data) -> dict:
        cfg = self.config
        obs, mask, amr_ids = build_cluster_observation(
            cluster, worker_predictions, map_data, cfg)
        obs = obs.to(self.device)
        mask = mask.to(self.device)
        with torch.no_grad():
            if self.deterministic:
                actions, value = self.policy(obs, mask)
            else:
                actions, _, _, value = self.policy.act(obs, mask, deterministic=False)
        self.last_value = float(np.atleast_1d(value.detach().cpu().numpy()).ravel()[0])
        acts = actions[0].detach().cpu().numpy()

        out = {}
        for k, amr_id in enumerate(amr_ids):
            out[amr_id] = acts[k]
        # Clusters larger than max_agents: the overflow AMRs fall back to the
        # deterministic V0 proposal so they are still planned and shielded.
        for agent in cluster:
            if agent.id not in out:
                fwd = float(np.clip(agent.goal_s - agent.s, 0.0, cfg.max_forward_dist))
                out[agent.id] = np.array([fwd or cfg.max_forward_dist, 0.0, 1.0])
        return out
