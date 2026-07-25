"""Step E (spatial) -- cluster-level local PATH replanning.

Beyond V0/V1 (which only modulate speed along fixed rails), this layer lets the
AMRs of a conflict cluster temporarily LEAVE their rails and move freely in 2D
to route around workers/each other, then rejoin the reference path downstream.

Design (per the agreed spec):
- After Step D, each conflict cluster owns its hull region -> a BUSY AREA that is
  locked: non-member AMRs must wait outside until the cluster clears.
- Inside the region the member AMRs do holonomic 2D motion toward a downstream
  rejoin point (where their rail exits the region). Each step they head toward a
  local goal; a 2D safety SHIELD projects the move so it never enters a worker
  hard no-go lobe, never collides with a cluster peer.
- `RuleLocalReplanner` is the deterministic (V0-style) baseline: greedily head to
  the exit via sampled headings. The learned multi-agent attention policy (V1)
  plugs in by replacing the local-goal proposal, behind the same shield.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .prediction import PredictorConfig, IntentParticlePredictor
from .safety_inflation import SafetyInflationConfig, SafetyInflationModel
from .factory_map import GOALS, OBSTACLES
from .scenario import make_workers, make_amrs
from .geometry import point_in_polygon, safety_tube_polygon
from .affected_amr import (
    AMR, CentralizedPlanner, ConflictChecker, ConflictResult, amr_human_collision,
    T_REPLAN_DEFAULT, V_AMR_TYPICAL_DEFAULT,
)
from .conflict_cluster import ConflictClusterBuilder


@dataclass
class LocalReplanConfig:
    max_speed: float = 0.50            # m/s holonomic maneuver speed in region
    exit_tol: float = 0.60             # m: reached rejoin point (loose -> RL hits it more)
    exit_margin: float = 0.40          # m past the region boundary for the exit pt
    peer_clearance: float = 0.85       # m centre-to-centre between cluster AMRs
    worker_collision_dist: float = 0.55
    worker_danger: float = 0.95        # m: hard keep-out around a worker's position
    sample_angles_deg: tuple[float, ...] = (
        0, 20, -20, 40, -40, 60, -60, 90, -90, 120, -120, 150, -150, 180)
    rail_sample: float = 0.10          # m: arc-length step when finding the exit
    max_local_frames: int = 150        # force a rejoin if stuck this long (liveness)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def compute_exit_s(amr: AMR, hull: np.ndarray, cfg: LocalReplanConfig) -> float:
    """First arc-length downstream of ``progress`` where the rail leaves ``hull``."""
    s = amr.progress
    last_inside = s
    while s <= amr.total_length:
        if point_in_polygon(amr.position_at(s), hull):
            last_inside = s
            s += cfg.rail_sample
        else:
            break
    return float(min(last_inside + cfg.exit_margin, amr.total_length))


def _rot(vec: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c])


_SAMPLE_ANGLES = (0, 20, -20, 40, -40, 60, -60, 90, -90, 120, -120, 150, -150, 180)


def shielded_displacement(cur, pref_dir, step, worker_pos, peers, cfg):
    """Project a preferred move onto the 2D safe set (circular worker keep-out +
    peer clearance). Returns (new_xy, heading); holds in place if fully boxed.

    Shared by the rule replanner (pref = greedy-to-exit) and the RL replanner
    (pref = policy local-goal direction) -- safety is guaranteed identically.
    """
    keep = cfg.worker_collision_dist + 0.30
    n = float(np.linalg.norm(pref_dir))
    if n < 1e-9:
        return cur.copy(), None
    pref = pref_dir / n

    def safe(p):
        for wp in worker_pos:
            if float(np.linalg.norm(p - wp)) < keep:
                return False
        for q in peers:
            if float(np.linalg.norm(p - q)) < cfg.peer_clearance:
                return False
        return True

    best = None
    for ang in _SAMPLE_ANGLES:
        d = _rot(pref, ang)
        cand = cur + d * step
        if not safe(cand):
            continue
        wc = min((float(np.linalg.norm(cand - wp)) for wp in worker_pos), default=9.0)
        score = float(np.dot(d, pref)) + 0.5 * min(wc, 1.5)
        if best is None or score > best[0]:
            best = (score, cand, d)
    if best is None and worker_pos:
        nw = min(worker_pos, key=lambda wp: float(np.linalg.norm(cur - wp)))
        away = (cur - nw) / (float(np.linalg.norm(cur - nw)) + 1e-9)
        cand = cur + away * step
        if safe(cand):
            best = (0.0, cand, away)
    if best is None:
        return cur.copy(), None
    return best[1], math.atan2(best[2][1], best[2][0])


class ExternalReplanner:
    """Local replanner driven by externally supplied per-AMR goal directions
    (the RL policy). Call ``set_actions({name: desired_displacement})`` each
    frame before ``LocalReplanSim.step``; the displacement is shield-projected."""

    def __init__(self, cfg: LocalReplanConfig | None = None):
        self.cfg = cfg or LocalReplanConfig()
        self.pending: dict[str, np.ndarray] = {}
        self.last_headings: dict[str, float] = {}

    def set_actions(self, actions: dict):
        self.pending = actions

    def step(self, members, worker_pos, dt):
        cfg = self.cfg
        reserved: list[np.ndarray] = []
        reached = {}
        for amr in members:
            cur = amr.current_xy()
            goal = amr.position_at(amr.exit_s)
            dist = float(np.linalg.norm(goal - cur))
            if dist < cfg.exit_tol:
                reached[amr.name] = True; reserved.append(cur); continue
            reached[amr.name] = False
            desired = self.pending.get(amr.name)
            if desired is None:
                desired = (goal - cur)                       # fallback: head to exit
            step = min(cfg.max_speed * dt, float(np.linalg.norm(desired)) or cfg.max_speed * dt)
            new_xy, hd = shielded_displacement(cur, np.asarray(desired, float),
                                               step, worker_pos, reserved, cfg)
            amr.xy = new_xy
            if hd is not None:
                self.last_headings[amr.name] = hd
            reserved.append(amr.xy)
        return reached


# ---------------------------------------------------------------------------
# Rule-based local replanner (deterministic baseline)
# ---------------------------------------------------------------------------

class RuleLocalReplanner:
    """Greedy holonomic routing to the rejoin point, behind the 2D shield."""

    def __init__(self, cfg: LocalReplanConfig | None = None):
        self.cfg = cfg or LocalReplanConfig()
        self.last_headings: dict[str, float] = {}

    def _safe(self, p, worker_pos, peers, keep_out, cfg):
        for wp in worker_pos:
            if float(np.linalg.norm(p - wp)) < keep_out:
                return False
        for q in peers:
            if float(np.linalg.norm(p - q)) < cfg.peer_clearance:
                return False
        return True

    def step(self, members, worker_pos, dt):
        """Move each local-mode member one step (greedy-to-exit + worker evade,
        behind a circular keep-out shield). members ordered by priority."""
        cfg = self.cfg
        keep = cfg.worker_collision_dist + 0.30          # ~0.85 m hard keep-out
        evade = 1.5
        reserved: list[np.ndarray] = []
        reached = {}
        for amr in members:
            cur = amr.current_xy()
            goal = amr.position_at(amr.exit_s)
            to_goal = goal - cur
            dist = float(np.linalg.norm(to_goal))
            if dist < cfg.exit_tol:
                reached[amr.name] = True; reserved.append(cur); continue
            reached[amr.name] = False
            gdir = to_goal / (dist + 1e-9)
            # nearest worker -> evade bias
            nw = None
            for wp in worker_pos:
                dd = float(np.linalg.norm(cur - wp))
                if nw is None or dd < nw[0]:
                    nw = (dd, wp)
            if nw is not None and nw[0] < evade:
                away = (cur - nw[1]) / (nw[0] + 1e-9)
                pref = gdir * 0.4 + away * 1.0
                pref = pref / (np.linalg.norm(pref) + 1e-9)
            else:
                pref = gdir
            step = min(cfg.max_speed * dt, dist)
            best = None
            for ang in cfg.sample_angles_deg:
                d = _rot(pref, ang)
                cand = cur + d * step
                if not self._safe(cand, worker_pos, reserved, keep, cfg):
                    continue
                wc = min((float(np.linalg.norm(cand - wp)) for wp in worker_pos), default=9.0)
                score = float(np.dot(d, gdir)) + 0.5 * min(wc, 1.5)
                if best is None or score > best[0]:
                    best = (score, cand, d)
            if best is None and nw is not None:
                # emergency: flee directly away from the nearest worker
                away = (cur - nw[1]) / (nw[0] + 1e-9)
                cand = cur + away * step
                if self._safe(cand, worker_pos, reserved, cfg.worker_collision_dist + 0.05, cfg):
                    best = (0.0, cand, away)
            if best is not None:
                amr.xy = best[1]
                self.last_headings[amr.name] = math.atan2(best[2][1], best[2][0])
                reserved.append(amr.xy)
            else:
                reserved.append(cur)
        return reached


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class _Lock:
    __slots__ = ("members", "hull", "color")
    def __init__(self, members, hull, color):
        self.members = set(members); self.hull = hull; self.color = color


class LocalReplanSim:
    """Full A->D pipeline + spatial local-path-replanning with busy-area locks."""

    def __init__(self, num_frames=360, num_workers=2, num_amrs=6, seed=7,
                 cfg: LocalReplanConfig | None = None, replanner=None,
                 worker_frames=None):
        self.cfg = cfg or LocalReplanConfig()
        self._wf = worker_frames           # optional precomputed per-frame worker_data
        self.num_frames = num_frames
        self.pcfg = PredictorConfig(seed=seed)
        self.dt = self.pcfg.dt
        self.horizon_T = self.pcfg.horizon_steps
        self.safety = SafetyInflationModel(SafetyInflationConfig())
        self.workers = make_workers(num_frames, self.dt, num_workers)
        self.predictors = [IntentParticlePredictor(GOALS, OBSTACLES, self.pcfg,
                           rng=np.random.default_rng(seed + 17 * i))
                           for i in range(len(self.workers))]
        self.amrs = make_amrs(num_amrs)
        self.planner = CentralizedPlanner(dt=self.dt)
        self.cluster_builder = ConflictClusterBuilder()
        self.replanner = replanner or RuleLocalReplanner(self.cfg)
        self.locks: list[_Lock] = []
        self._local_start: dict[str, int] = {}

    def _worker_data(self, frame):
        if self._wf is not None:
            return self._wf[frame]
        wd = []
        for w, predictor in zip(self.workers, self.predictors):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            out = predictor.rollout(obs)
            hard, soft, _ = self.safety.inflate_all(
                out["ellipses"], self.dt, out["belief"], mean_traj=out["mean"])
            wd.append(dict(name=w["name"], inflated=hard, soft=soft,
                           centers=out["ellipses"][:, :2]))
        return wd

    def _in_any_lock(self, xy, exclude_member=None):
        for lk in self.locks:
            if exclude_member in lk.members:
                continue
            if point_in_polygon(xy, lk.hull):
                return lk
        return False

    def step(self, frame):
        cfg = self.cfg
        wd = self._worker_data(frame)
        worker_hard0 = [w["inflated"][0] for w in wd]
        worker_pos = [(w["truth"][frame] if frame < len(w["truth"]) else w["truth"][-1]) for w in self.workers]

        rail_active = [a for a in self.amrs if a.is_active(frame) and not a.local_mode]

        # --- Steps A-D on rail-mode AMRs ---
        self.planner.resolve(rail_active, frame)
        results: dict[str, ConflictResult] = {}
        for a in rail_active:
            results[a.name] = ConflictChecker.check(a, self.dt, self.horizon_T, wd,
                                                    t_replan=T_REPLAN_DEFAULT,
                                                    v_amr_typical=V_AMR_TYPICAL_DEFAULT)
        cluster_result = self.cluster_builder.build(rail_active, results, frame,
                                                    self.dt, self.horizon_T)

        # --- form busy areas: put new clusters into local mode ---
        self.newly_formed = []           # scenarios harvested this frame (for RL)
        locked_names = {n for lk in self.locks for n in lk.members}
        for cl in cluster_result.clusters:
            if any(n in locked_names for n in cl.member_names):
                continue
            members = [a for a in self.amrs if a.name in cl.member_names]
            for a in members:
                a.enter_local(compute_exit_s(a, cl.hull, cfg))
                self._local_start[a.name] = frame
            self.locks.append(_Lock(cl.member_names, cl.hull, cl.color))
            locked_names.update(cl.member_names)
            self.newly_formed.append(dict(
                frame=frame, hull=cl.hull,
                members=[dict(name=a.name, color=a.color,
                              entry=a.current_xy().copy(),
                              exit=a.position_at(a.exit_s).copy())
                         for a in members]))

        # --- move local-mode AMRs (priority: nearest to exit first) ---
        for lk in self.locks:
            members = [a for a in self.amrs if a.name in lk.members and a.local_mode]
            members.sort(key=lambda a: float(np.linalg.norm(a.current_xy() - a.position_at(a.exit_s))))
            reached = self.replanner.step(members, worker_pos, self.dt)
            for a in members:
                # worker collision in 2D (uses current_xy)
                p = a.current_xy()
                for w in self.workers:
                    wp = w["truth"][frame] if frame < len(w["truth"]) else w["truth"][-1]
                    if float(np.linalg.norm(p - wp)) < cfg.worker_collision_dist:
                        a.mark_collision(frame, w["name"]); a.xy = p
                        break
                if a.local_mode and reached.get(a.name):
                    a.resume_rail()
                elif (a.local_mode and not a.collided
                      and frame - self._local_start.get(a.name, frame) > cfg.max_local_frames):
                    a.resume_rail()          # liveness: stuck too long -> force rejoin

        # --- release locks whose members all rejoined the rail ---
        self.locks = [lk for lk in self.locks
                      if any((a.local_mode and not a.collided)
                             for a in self.amrs if a.name in lk.members)]

        # --- advance rail-mode AMRs, holding at locked-area boundaries ---
        for a in rail_active:
            if not a.is_active(frame) or a.local_mode:
                continue
            nxt = a.position_at(min(a.progress + a.actual_speed * self.dt, a.granted_s))
            if self._in_any_lock(nxt, exclude_member=a.name):
                a.actual_speed = 0.0           # wait outside the busy area
                a.waiting_for = "busy-area"
            else:
                hit, hw = amr_human_collision(a, frame, self.workers, cfg.worker_collision_dist)
                if hit:
                    a.mark_collision(frame, hw)
                else:
                    a.step(self.dt)

        return dict(worker_data=wd, results=results, clusters=cluster_result,
                    locks=list(self.locks))
