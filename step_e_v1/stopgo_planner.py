"""Stop-and-go baseline: drive or halt, nothing in between.

The classic industrial AMR safety controller. It keeps the whole Step-E
machinery -- the same observation of the scene, the same TTC priority order,
the same space-time reservation shield, the same commands -- and removes only
the *action set*:

    candidates = { GO   : full planner-allowed speed, on the reference path,
                   STOP : hold position }

so it has

    * no speed modulation   (V0's 0.66 / 0.33 factors are gone),
    * no lateral shift or detour  (goal_lat is pinned to 0),
    * no emergency reverse,
    * no least-unsafe fallback -- when nothing is safe it simply stops and
      waits, which is what "stop and go" means.

Comparing it against V0 isolates exactly what continuous speed control plus the
lateral degree of freedom are worth, and comparing it against V1 shows what the
learned proposal adds on top of that.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .planner_base import ShieldedReplanner


class StopAndGoReplanner(ShieldedReplanner):
    name = "STOP-GO"
    prefer_proposal = False

    def __init__(self, config):
        # The shield's discretionary escapes are part of the planner's policy,
        # not of the safety model, so a stop-and-go controller must not have
        # them: it stops and waits instead.
        super().__init__(replace(config,
                                 allow_emergency_reverse=False,
                                 least_unsafe_fallback=False,
                                 use_backup_candidates=False,
                                 use_detour_candidates=False,
                                 candidate_grid=False,
                                 lazy_backups=False))

    def propose_actions(self, cluster, worker_predictions, map_data) -> dict:
        cfg = self.config
        out = {}
        for agent in cluster:
            fwd = float(np.clip(agent.goal_s - agent.s, 0.0, cfg.max_forward_dist))
            out[agent.id] = np.array([fwd or cfg.max_forward_dist, 0.0, 1.0])
        return out

    def candidate_actions(self, agent, a0):
        """GO or STOP -- no slow-down, no shift, no detour."""
        go = np.array([a0[0], 0.0, 1.0])
        stop = np.array([a0[0], 0.0, 0.0])
        return [go, stop]
