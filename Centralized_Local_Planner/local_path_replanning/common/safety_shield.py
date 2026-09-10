"""Space-time reservation safety shield -- shared by every method.

Every candidate trajectory must pass four checks:

    1. AMR-worker  : stay out of the Step-B predicted no-go tube, time-aligned
    2. AMR-AMR     : keep ``min_amr_distance`` from every already reserved
                     trajectory at the same look-ahead time
    3. static map  : stay out of the workstation rectangles inflated by the
                     AMR footprint
    4. feasibility : respect v_max / a_max / omega_max / curvature_max

Safe candidates are ranked by the cost

    J = w1 J_worker + w2 J_amr + w3 J_route + w4 J_smooth + w5 J_delay

and the cheapest one is executed. If no candidate is safe the AMR STOPs, and
the STOP trajectory is reserved so lower-priority AMRs plan around it.

The policy proposes; this module decides what may be executed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_INF = float("inf")


# ---------------------------------------------------------------------------
# Vectorised geometry
# ---------------------------------------------------------------------------

def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray-cast test, many points against one polygon. (M,2),(K,2) -> (M,) bool."""
    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    xi, yi = poly[:, 0][None, :], poly[:, 1][None, :]
    xj = np.roll(poly[:, 0], 1)[None, :]
    yj = np.roll(poly[:, 1], 1)[None, :]
    cond = ((yi > y) != (yj > y))
    xint = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
    return (np.logical_and(cond, x < xint).sum(axis=1) % 2).astype(bool)


def points_polygon_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to the polygon boundary. (M,2),(K,2) -> (M,)."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a                                        # (K,2)
    ap = points[:, None, :] - a[None, :, :]           # (M,K,2)
    denom = (ab ** 2).sum(axis=1) + 1e-12             # (K,)
    t = np.clip((ap * ab[None, :, :]).sum(axis=2) / denom, 0.0, 1.0)   # (M,K)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(points[:, None, :] - proj, axis=2).min(axis=1)


def points_in_polygons(points: np.ndarray, polys: np.ndarray) -> np.ndarray:
    """Paired test: point m against polygon m. (M,2),(M,K,2) -> (M,) bool."""
    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    xi, yi = polys[:, :, 0], polys[:, :, 1]
    xj, yj = np.roll(xi, 1, axis=1), np.roll(yi, 1, axis=1)
    cond = (yi > y) != (yj > y)
    xint = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
    return (np.logical_and(cond, x < xint).sum(axis=1) % 2).astype(bool)


def points_polygons_distance(points: np.ndarray, polys: np.ndarray) -> np.ndarray:
    """Paired distance: point m to polygon m boundary. (M,2),(M,K,2) -> (M,)."""
    a = polys
    b = np.roll(polys, -1, axis=1)
    ab = b - a                                          # (M,K,2)
    ap = points[:, None, :] - a                         # (M,K,2)
    denom = (ab ** 2).sum(axis=2) + 1e-12               # (M,K)
    tt = np.clip((ap * ab).sum(axis=2) / denom, 0.0, 1.0)
    proj = a + tt[:, :, None] * ab
    return np.linalg.norm(points[:, None, :] - proj, axis=2).min(axis=1)


def points_in_rect(points: np.ndarray, rect, margin: float) -> np.ndarray:
    x, y, w, h = rect
    return ((points[:, 0] > x - margin) & (points[:, 0] < x + w + margin)
            & (points[:, 1] > y - margin) & (points[:, 1] < y + h + margin))


# ---------------------------------------------------------------------------
# Reservation table
# ---------------------------------------------------------------------------

class SpaceTimeReservation:
    """Trajectories already granted to higher-priority AMRs this cycle."""

    def __init__(self, min_distance: float):
        self.min_distance = float(min_distance)
        self.entries: list[tuple[str, np.ndarray]] = []      # (id, positions)

    def reserve(self, amr_id: str, positions: np.ndarray) -> None:
        self.entries.append((amr_id, np.asarray(positions, dtype=float)))

    def min_distance_to(self, positions: np.ndarray, skip_id: str | None = None) -> float:
        best = _INF
        for other_id, other in self.entries:
            if skip_id is not None and other_id == skip_id:
                continue
            n = min(len(positions), len(other))
            if n == 0:
                continue
            d = float(np.linalg.norm(positions[:n] - other[:n], axis=1).min())
            best = min(best, d)
        return best

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# The four checks
# ---------------------------------------------------------------------------

@dataclass
class CheckReport:
    safe: bool = True
    reason: str = ""
    hit_worker: bool = False
    soft_worker_fraction: float = 0.0   # tube overlap beyond the hard window
    final_worker_clearance: float = _INF  # clearance at the END of the horizon
    hit_amr: bool = False
    hit_static: bool = False
    hit_dynamic: bool = False
    min_worker_clearance: float = _INF
    min_amr_distance: float = _INF
    min_obstacle_distance: float = _INF
    first_violation_time: float = _INF


def check_worker_collision(traj, worker_predictions, config,
                           stride: int | None = None) -> CheckReport:
    """AMR footprint vs every worker's predicted no-go tube, time-aligned.

    Inside ``config.hard_horizon_sec`` a tube overlap makes the candidate
    infeasible. Beyond it the overlap is recorded as a soft violation that only
    adds cost: enforcing the whole 5 s prediction as a hard constraint is what
    freezes the AMR in the middle of an aisle, and a frozen AMR is exactly what
    an oblivious worker then walks into.
    """
    rep = CheckReport()
    if not worker_predictions:
        return rep
    stride = stride or config.shield_stride
    idx = np.arange(0, len(traj.times), max(int(stride), 1))
    if idx[-1] != len(traj.times) - 1:
        idx = np.append(idx, len(traj.times) - 1)
    pts = traj.positions[idx]
    times = traj.times[idx]
    within_hard = times <= config.hard_horizon_sec + 1e-9

    for w in worker_predictions:
        # Time-align every sample with the prediction step of the same
        # look-ahead time, then test all samples in one vectorised pass.
        j = np.clip(np.round(times / max(w.dt, 1e-9)).astype(int) - 1, 0, w.horizon - 1)
        j[times <= 1e-9] = 0
        centre_d = np.linalg.norm(pts - w.centers[j], axis=1)
        rep.min_worker_clearance = min(rep.min_worker_clearance, float(centre_d.min()))
        rep.final_worker_clearance = min(rep.final_worker_clearance, float(centre_d[-1]))
        hit = centre_d < config.worker_circle_clearance
        if config.use_polygon_tube:
            polys = w.hard_lobes[j][:, ::config.polygon_stride, :]
            hit |= points_in_polygons(pts, polys)
            hit |= points_polygons_distance(pts, polys) < config.amr_radius

        soft_hit = hit & ~within_hard
        if soft_hit.size:
            rep.soft_worker_fraction = max(rep.soft_worker_fraction,
                                           float(soft_hit.sum()) / float(soft_hit.size))
        hard_hit = hit & within_hard
        if bool(hard_hit.any()):
            k = int(np.argmax(hard_hit))
            rep.hit_worker = True
            if rep.safe:
                rep.safe = False
                rep.reason = f"worker:{w.name}@{float(times[k]):.1f}s"
                rep.first_violation_time = float(times[k])
    return rep


def check_amr_amr_collision(traj, reservation: SpaceTimeReservation, config,
                            skip_id: str | None = None) -> CheckReport:
    rep = CheckReport()
    if reservation is None or len(reservation) == 0:
        return rep
    d = reservation.min_distance_to(traj.positions, skip_id=skip_id)
    rep.min_amr_distance = d
    if d < config.min_amr_distance:
        rep.safe = False
        rep.hit_amr = True
        rep.reason = "amr-amr"
    return rep


def check_static_obstacle_collision(traj, map_data, config) -> CheckReport:
    rep = CheckReport()
    if map_data is None or not map_data.obstacles:
        return rep
    margin = config.amr_radius + config.obstacle_margin
    pts = traj.positions
    for rect in map_data.obstacles:
        inside = points_in_rect(pts, rect, margin)
        if bool(inside.any()):
            rep.safe = False
            rep.hit_static = True
            rep.reason = "static-obstacle"
            rep.first_violation_time = float(traj.times[int(np.argmax(inside))])
            rep.min_obstacle_distance = 0.0
            return rep
    return rep


def check_dynamic_feasibility(traj, config) -> CheckReport:
    """Speed / acceleration limits plus the analytic manoeuvre geometry.

    The lateral deviation is what the local planner actually commands, so the
    kinematic limits are applied to the MANOEUVRE (lateral slope, lateral
    curvature and the resulting yaw rate), not to the absolute heading trace:
    the global reference path has 90-degree corners the fleet already drives,
    and charging them to the local planner would reject every trajectory.
    """
    rep = CheckReport()
    dt = float(traj.times[1] - traj.times[0]) if len(traj.times) > 1 else config.dt
    v = traj.speeds

    def fail(reason):
        rep.safe = False
        rep.hit_dynamic = True
        rep.reason = reason
        return rep

    if float(v.max()) > config.v_max * 1.05 + 1e-6:
        return fail("v_max")
    if len(v) > 1:
        acc = np.abs(np.diff(v)) / max(dt, 1e-9)
        if float(acc.max()) > config.a_max * 1.05 + 1e-6:
            return fail("a_max")
    if traj.lat_speed_max > config.v_lat_max * 1.05 + 1e-6:
        return fail("v_lat_max")
    return rep


def check_trajectory(traj, worker_predictions, reservation, map_data, config,
                     skip_id: str | None = None) -> CheckReport:
    """Run all four checks; returns the first failure with its diagnostics."""
    out = CheckReport()
    for rep in (check_dynamic_feasibility(traj, config),
                check_static_obstacle_collision(traj, map_data, config),
                check_worker_collision(traj, worker_predictions, config),
                check_amr_amr_collision(traj, reservation, config, skip_id)):
        out.min_worker_clearance = min(out.min_worker_clearance, rep.min_worker_clearance)
        out.min_amr_distance = min(out.min_amr_distance, rep.min_amr_distance)
        out.min_obstacle_distance = min(out.min_obstacle_distance, rep.min_obstacle_distance)
        out.hit_worker |= rep.hit_worker
        out.soft_worker_fraction = max(out.soft_worker_fraction, rep.soft_worker_fraction)
        out.final_worker_clearance = min(out.final_worker_clearance, rep.final_worker_clearance)
        out.hit_amr |= rep.hit_amr
        out.hit_static |= rep.hit_static
        out.hit_dynamic |= rep.hit_dynamic
        if not rep.safe and out.safe:
            out.safe = False
            out.reason = rep.reason
            out.first_violation_time = rep.first_violation_time
    return out


# ---------------------------------------------------------------------------
# Candidate cost  J = w1 Jw + w2 Ja + w3 Jroute + w4 Jsmooth + w5 Jdelay
# ---------------------------------------------------------------------------

def trajectory_cost(agent, traj, report: CheckReport, config) -> dict:
    d_ref_w = 1.5
    j_worker = float(np.clip(1.0 - report.min_worker_clearance / d_ref_w, 0.0, 1.0) ** 2)
    j_worker_soft = float(report.soft_worker_fraction)
    d_ref_a = config.min_amr_distance * 1.5
    j_amr = float(np.clip(1.0 - report.min_amr_distance / d_ref_a, 0.0, 1.0) ** 2)
    j_route = float(np.abs(traj.lateral).mean() / max(config.max_lateral_offset, 1e-9))
    dv = np.abs(np.diff(traj.speeds)).mean() if len(traj.speeds) > 1 else 0.0
    dh = np.diff(traj.headings) if len(traj.headings) > 1 else np.zeros(1)
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    j_smooth = float(dv / max(config.v_max, 1e-9) + np.abs(dh).mean() / math.pi)
    j_delay = float(1.0 - np.clip(traj.speeds.mean() / max(config.v_max, 1e-9), 0.0, 1.0))
    total = (config.w_worker * j_worker + config.w_worker_soft * j_worker_soft
             + config.w_amr * j_amr
             + config.w_route * j_route + config.w_smooth * j_smooth
             + config.w_delay * j_delay)
    return dict(total=total, worker=j_worker, worker_soft=j_worker_soft,
                amr=j_amr, route=j_route, smooth=j_smooth, delay=j_delay)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass
class ShieldResult:
    amr_id: str
    action: np.ndarray
    trajectory: object
    cost: float = _INF
    safe: bool = True
    modified: bool = False              # shield did not execute the proposal
    candidate_index: int = 0
    expanded: int = 0                   # candidates actually rolled out
    reason: str = ""
    report: CheckReport = field(default_factory=CheckReport)
    cost_terms: dict = field(default_factory=dict)


def action_distance(a, b, config) -> float:
    """Normalised distance between two decoded actions."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.linalg.norm([
        (a[0] - b[0]) / max(config.max_forward_dist, 1e-9),
        (a[1] - b[1]) / max(config.max_lateral_offset, 1e-9),
        a[2] - b[2],
    ]))


def select_safe_trajectory(agent, candidates, worker_predictions, reservation,
                           map_data, config, prefer_proposal: bool = False,
                           prev_action=None) -> ShieldResult:
    """Choose the trajectory this AMR may execute.

    ``candidates`` is an ordered list of ``(action, Trajectory)``; index 0 is
    the proposal, the rest are the specification's backups.

    ``prefer_proposal=False`` (the rule methods): accept the lowest-cost SAFE candidate.
    ``prefer_proposal=True``  (the learned one): execute the proposal whenever it is safe,
    and only fall back to the cheapest safe backup when it is not -- so the
    learned policy owns efficiency and ``shield_modified`` means a genuine
    override.

    If nothing is safe the AMR STOPs (an explicitly built stop trajectory).
    """
    from .trajectory_rollout import rollout_action, stop_action

    best: ShieldResult | None = None
    evaluated: list[ShieldResult] = []

    # Lazy candidate set (see PlannerConfig.lazy_backups): check the proposal on its
    # own first and only materialise the backups if it fails.
    lazy = getattr(candidates, "first", None)
    if lazy is not None and prefer_proposal:
        action, traj = lazy
        rep = check_trajectory(traj, worker_predictions, reservation, map_data,
                               config, skip_id=agent.id)
        if rep.safe:
            terms = trajectory_cost(agent, traj, rep, config)
            if prev_action is not None:
                terms["switch"] = action_distance(action, prev_action, config)
                terms["total"] += config.w_switch * terms["switch"]
            return ShieldResult(amr_id=agent.id, action=np.asarray(action, float),
                                trajectory=traj, cost=terms["total"], safe=True,
                                modified=False, candidate_index=0, expanded=1,
                                reason="", report=rep, cost_terms=terms)
    if callable(candidates):
        candidates = candidates()
    for i, (action, traj) in enumerate(candidates):
        rep = check_trajectory(traj, worker_predictions, reservation, map_data,
                               config, skip_id=agent.id)
        terms = trajectory_cost(agent, traj, rep, config)
        if prev_action is not None:
            terms["switch"] = action_distance(action, prev_action, config)
            terms["total"] += config.w_switch * terms["switch"]
        res = ShieldResult(amr_id=agent.id, action=np.asarray(action, float),
                           trajectory=traj, cost=terms["total"], safe=rep.safe,
                           modified=(i != 0), candidate_index=i,
                           reason="" if rep.safe else rep.reason,
                           report=rep, cost_terms=terms)
        evaluated.append(res)
        if not rep.safe:
            continue
        if i == 0 and prefer_proposal:
            return res
        if best is None or terms["total"] < best.cost:
            best = res
    if best is not None:
        return best

    # Nothing is safe. Standing still is NOT automatically the best answer:
    # when a worker walks into a halted AMR, holding position is what causes
    # the collision. Fall back to the least-unsafe candidate -- no static or
    # AMR violation first, then the largest worker clearance. The STOP
    # candidate is evaluated alongside the others, so it still wins whenever
    # moving would be worse.
    from .action_decoder import reverse_action

    emergency = [stop_action(config)]
    if config.allow_emergency_reverse:
        emergency.append(reverse_action(agent, config))
    for k, act in enumerate(emergency):
        traj = rollout_action(agent, act, config.horizon_sec, config.dt,
                              config, map_data)
        rep = check_trajectory(traj, worker_predictions, reservation, map_data,
                               config, skip_id=agent.id)
        terms = trajectory_cost(agent, traj, rep, config)
        if prev_action is not None:
            terms["switch"] = action_distance(act, prev_action, config)
            terms["total"] += config.w_switch * terms["switch"]
        evaluated.append(ShieldResult(
            amr_id=agent.id, action=np.asarray(act, float), trajectory=traj,
            cost=terms["total"], safe=False, modified=True, candidate_index=-(k + 1),
            reason=rep.reason or "no-safe-candidate", report=rep, cost_terms=terms))

    def severity(r: ShieldResult):
        # Rank by how the separation EVOLVES, not just by its minimum: with an
        # approaching worker a halted AMR keeps the best instantaneous
        # clearance yet the worst final one, so a pure min-clearance rule
        # freezes it in the worker's path. Weighting the end-of-horizon
        # clearance equally makes the evasive manoeuvre win, while standing
        # still still wins whenever moving would genuinely be worse.
        rep = r.report
        final = rep.final_worker_clearance if np.isfinite(rep.final_worker_clearance) else 9.9
        mind = rep.min_worker_clearance if np.isfinite(rep.min_worker_clearance) else 9.9
        return (int(rep.hit_dynamic), int(rep.hit_static), int(rep.hit_amr),
                -(0.5 * mind + 0.5 * final), r.cost)

    if config.least_unsafe_fallback:
        chosen = min(evaluated, key=severity)
    else:
        chosen = evaluated[-len(emergency)]        # plain STOP, as specified
    chosen.safe = False
    chosen.modified = True
    if not chosen.reason:
        chosen.reason = "no-safe-candidate"
    return chosen


def select_safe_joint_trajectories(cluster, candidate_trajs, worker_predictions,
                                   map_data, config, order=None,
                                   prefer_proposal: bool = False,
                                   prev_actions: dict | None = None) -> dict:
    """Sequentially shield a whole cluster, reserving as we go.

    ``candidate_trajs`` maps an AMR id to its ordered ``(action, Trajectory)``
    candidate list (a single pair is accepted and treated as a one-element
    list). ``order`` is the planning priority (default: the cluster order).
    Returns ``{amr_id: ShieldResult}``.
    """
    agents = {a.id: a for a in cluster}
    ids = list(order) if order is not None else [a.id for a in cluster]
    reservation = SpaceTimeReservation(config.min_amr_distance)
    out: dict[str, ShieldResult] = {}
    for amr_id in ids:
        cand = candidate_trajs.get(amr_id)
        if cand is None:
            continue
        if isinstance(cand, tuple):
            cand = [cand]
        res = select_safe_trajectory(agents[amr_id], cand, worker_predictions,
                                     reservation, map_data, config,
                                     prefer_proposal=prefer_proposal,
                                     prev_action=(prev_actions or {}).get(amr_id))
        reservation.reserve(amr_id, res.trajectory.positions)
        out[amr_id] = res
    return out
