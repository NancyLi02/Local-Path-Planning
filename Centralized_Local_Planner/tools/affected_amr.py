"""Step C -- affected-AMR identification.

AMR fleet model + centralized QR-cell planner + space-time conflict check
against worker safety tubes (AMR, CentralizedPlanner, ConflictChecker)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import point_in_polygon, polygon_signed_radius
from .safety_inflation import SafetyInflationModel


# TTC threshold (at full AMR speed) above which graceful slowdown suffices.
# Scaled with the slower fleet speed so the spatial lookahead (~1.0 m at
# v_amr * t_replan) stays comparable to the faster demo (0.65 m/s * 1.5 s).
T_REPLAN_DEFAULT: float = 2.8

# Nominal fleet speed -- every AMR uses this commanded speed.
AMR_SPEED_DEFAULT: float = 0.35          # m/s  (~1.3 km/h, cautious factory pace)

# Reference AMR speed for alpha_amr velocity gating (matches fleet nominal).
V_AMR_TYPICAL_DEFAULT: float = AMR_SPEED_DEFAULT

class AMR:
    """One AMR. Traverses its reference path EXACTLY ONCE (no looping).

    The floor is densely covered with QR codes (~``qr_spacing`` apart), so the
    reference path is discretised into many intermediate QR waypoints. The AMR
    is only ever cleared to drive up to ``granted_s`` (the arc-length of the
    last QR waypoint granted by the CentralizedPlanner). On arriving there it
    reports to the planner and requests the next waypoint; ``request_frame``
    records WHEN it arrived, which is what the planner uses for first-come-
    first-served arbitration.
    """

    def __init__(
        self,
        name: str,
        color: str,
        waypoints: np.ndarray,
        commanded_speed: float = AMR_SPEED_DEFAULT,
        spawn_frame: int = 0,
        init_progress: float = 0.0,
        qr_spacing: float = 1.0,
    ):
        self.name = name
        self.color = color
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.commanded_speed = float(commanded_speed)
        self.spawn_frame = int(spawn_frame)
        self.progress = float(init_progress)
        self.actual_speed = float(commanded_speed)
        self.waiting_for: str = ""

        # Sticky collision state. Once an AMR collides with a human it is
        # considered disabled: it never moves again, its body is frozen at
        # the collision point, and downstream planners must route around it.
        self.collided: bool = False
        self.collision_frame: int = -1
        self.collision_worker: str = ""
        self.collision_pos: np.ndarray = np.zeros(2)
        self.collision_heading: float = 0.0
        # Gantt-row snapshot at the moment of collision. Once latched, the
        # row is frozen at this pattern for the rest of the simulation so
        # the user sees a post-mortem of the conflict that caused the crash.
        self.frozen_hard_mask: np.ndarray | None = None
        self.frozen_soft_mask: np.ndarray | None = None
        self.frozen_alpha_amr: float = 0.0
        self.frozen_t_replan_eff: float = 0.0

        diffs = np.diff(self.waypoints, axis=0)
        self._seg_lens = np.linalg.norm(diffs, axis=1)
        self._cum_lens = np.concatenate([[0.0], np.cumsum(self._seg_lens)])
        self.total_length = float(self._cum_lens[-1])

        # ---- dense QR waypoint grid along the reference path -------------
        self.qr_spacing = float(qr_spacing)
        self.qr_arclen = self._densify_arclens(self.qr_spacing)
        self.qr_points = np.array([self.position_at(s) for s in self.qr_arclen])
        # Arc-length the planner has cleared us to drive to. Starts at 0:
        # even the first move must be granted by the planner.
        self.granted_s: float = float(init_progress)
        # Frame at which we arrived at the current QR waypoint and sent the
        # "arrived, request next" message. None while en-route.
        self.request_frame: int | None = None

        # ---- Step E spatial local-replanning state -----------------------
        # While in a conflict cluster the AMR temporarily leaves its rail and
        # moves freely in 2D (holonomic) toward a downstream rejoin point. In
        # local mode ``xy`` is the true world position and ``progress`` is
        # frozen at the entry arc-length; on reaching ``exit_s`` it snaps back.
        self.local_mode: bool = False
        self.xy: np.ndarray | None = None
        self.exit_s: float | None = None

    def _densify_arclens(self, spacing: float) -> np.ndarray:
        """Arc-length positions of all QR waypoints (corners always kept)."""
        s_vals: list[float] = [0.0]
        for k in range(len(self._seg_lens)):
            seg = float(self._seg_lens[k])
            if seg < 1e-9:
                continue
            n = max(int(math.ceil(seg / max(spacing, 1e-6))), 1)
            base = float(self._cum_lens[k])
            for i in range(1, n + 1):
                s_vals.append(base + seg * (i / n))
        return np.unique(np.round(np.asarray(s_vals), 9))

    # ---- QR bookkeeping --------------------------------------------------

    def current_qr_index(self) -> int:
        """Index of the last QR waypoint already reached."""
        return int(np.searchsorted(self.qr_arclen, self.progress + 1e-6) - 1)

    def next_qr_index(self) -> int | None:
        """Index of the first QR waypoint beyond the granted horizon."""
        i = int(np.searchsorted(self.qr_arclen, self.granted_s + 1e-6))
        return i if i < len(self.qr_arclen) else None

    def at_granted_qr(self) -> bool:
        """True when the AMR has used up its clearance and sits at a QR."""
        return self.progress >= self.granted_s - 1e-6

    # ---- lifecycle ------------------------------------------------------

    def is_spawned(self, frame: int) -> bool:
        return frame >= self.spawn_frame

    def is_done(self) -> bool:
        return self.progress >= self.total_length - 1e-3

    def is_active(self, frame: int) -> bool:
        """Spawned, not finished, and not disabled by a collision."""
        return (
            self.is_spawned(frame)
            and not self.is_done()
            and not self.collided
        )

    def mark_collision(
        self,
        frame: int,
        worker_name: str,
        hard_mask: np.ndarray | None = None,
        soft_mask: np.ndarray | None = None,
        alpha_amr: float = 0.0,
        t_replan_eff: float = 0.0,
    ) -> None:
        """Latch the AMR into a permanent disabled / frozen state."""
        if self.collided:
            return
        self.collided = True
        self.collision_frame = frame
        self.collision_worker = worker_name
        self.collision_pos = self.position_at(self.progress)
        self.collision_heading = self.heading_at(self.progress)
        self.actual_speed = 0.0
        if hard_mask is not None:
            self.frozen_hard_mask = hard_mask.copy()
        if soft_mask is not None:
            self.frozen_soft_mask = soft_mask.copy()
        self.frozen_alpha_amr = float(alpha_amr)
        self.frozen_t_replan_eff = float(t_replan_eff)

    # ---- kinematics -----------------------------------------------------

    def position_at(self, s: float) -> np.ndarray:
        if self.total_length < 1e-9:
            return self.waypoints[0].copy()
        s = float(np.clip(s, 0.0, self.total_length))
        idx = int(np.searchsorted(self._cum_lens, s, side="right") - 1)
        idx = max(0, min(idx, len(self.waypoints) - 2))
        seg_len = max(self._seg_lens[idx], 1e-9)
        local = (s - self._cum_lens[idx]) / seg_len
        return self.waypoints[idx] * (1.0 - local) + self.waypoints[idx + 1] * local

    def heading_at(self, s: float) -> float:
        p1 = self.position_at(s)
        p2 = self.position_at(min(s + 0.05, self.total_length))
        d = p2 - p1
        if np.linalg.norm(d) < 1e-6:
            d = self.waypoints[-1] - self.waypoints[-2]
        return math.atan2(float(d[1]), float(d[0]))

    def position_after(self, future_dt: float, speed: float | None = None) -> np.ndarray:
        v = self.actual_speed if speed is None else speed
        return self.position_at(self.progress + v * future_dt)

    def predicted_positions(self, dt: float, T: int) -> np.ndarray:
        """Predictions over [dt, 2 dt, ..., T dt] using *actual* speed."""
        return np.array([self.position_after((t + 1) * dt) for t in range(T)])

    def step(self, dt: float) -> None:
        """Advance along the rail, but never beyond the planner's clearance."""
        self.progress = min(
            self.progress + self.actual_speed * dt,
            self.granted_s,
            self.total_length,
        )

    # ---- Step E spatial local-replanning -------------------------------
    def current_xy(self) -> np.ndarray:
        """World position: free 2D point in local mode, else rail position."""
        if self.local_mode and self.xy is not None:
            return self.xy.copy()
        return self.position_at(self.progress)

    def enter_local(self, exit_s: float) -> None:
        """Leave the rail; start free 2D motion toward downstream ``exit_s``."""
        self.local_mode = True
        self.xy = self.position_at(self.progress)
        self.exit_s = float(min(exit_s, self.total_length))

    def resume_rail(self) -> None:
        """Snap back onto the reference path at the rejoin arc-length."""
        if self.exit_s is not None:
            self.progress = float(min(self.exit_s, self.total_length))
            self.granted_s = max(self.granted_s, self.progress)
        self.local_mode = False
        self.xy = None
        self.exit_s = None


# ---------------------------------------------------------------------------
# Centralized planner (AMR-AMR yielding)
# ---------------------------------------------------------------------------


class CentralizedPlanner:
    """QR-cell reservation arbiter (waypoint-granting protocol).

    Protocol (mirrors the real QR-floor fleet):

    1.  An AMR may only drive up to its granted QR waypoint (``granted_s``).
    2.  On reaching it, the AMR reports "arrived" and requests the NEXT QR
        waypoint. ``request_frame`` latches the arrival frame.
    3.  Each planning cycle, all pending requests are served in
        first-come-first-served order of **waypoint-arrival time** (the frame
        the request was filed), NOT spawn order. Ties break on spawn frame.
    4.  A request is granted iff the target QR cell is free: no other AMR is
        physically within ``amr_safety_dist`` of it, no other AMR has been
        granted a cell within ``amr_safety_dist`` of it, and no DISABLED
        (collided) AMR is parked on it.
    5.  A denied AMR stops at its current QR waypoint (``actual_speed = 0``)
        and keeps its original request timestamp, so it does not lose its
        place in the queue while waiting.
    """

    def __init__(
        self,
        amr_safety_dist: float = 1.10,
        dt: float = 0.2,
        lateral_block_thresh: float = 0.7,
    ):
        self.amr_safety_dist = float(amr_safety_dist)
        self.dt = dt
        # A peer only blocks a requested cell if that cell lies on the peer's
        # line of travel (rear-end / crossing conflict). A peer merely passing
        # alongside on a parallel rail (lateral offset >= this threshold) does
        # NOT block -- otherwise two opposing lanes closer than amr_safety_dist
        # deadlock each other forever.
        self.lateral_block_thresh = float(lateral_block_thresh)

    def resolve(self, amrs: list[AMR], frame: int) -> None:
        active = [a for a in amrs if a.is_active(frame)]
        # Disabled AMRs permanently occupy their cell: others must wait.
        disabled = [a for a in amrs if a.collided]

        # --- 1) arrival bookkeeping --------------------------------------
        for a in active:
            if a.at_granted_qr():
                if a.next_qr_index() is None:
                    continue                      # fully granted to the end
                if a.request_frame is None:
                    a.request_frame = frame       # "arrived, request next"
            else:
                # En-route between QR cells: no pending request.
                a.request_frame = None
                a.waiting_for = ""
                a.actual_speed = a.commanded_speed

        # --- 2) serve requests FCFS by waypoint-arrival time -------------
        requesters = [a for a in active if a.request_frame is not None]
        requesters.sort(key=lambda a: (a.request_frame, a.spawn_frame))

        for a in requesters:
            nxt = a.next_qr_index()
            if nxt is None:
                a.request_frame = None
                a.waiting_for = ""
                continue
            target = a.qr_points[nxt]
            blocker = self._cell_blocker(a, target, active, disabled)
            if blocker is None:
                # GRANT: clearance extended one QR cell forward.
                a.granted_s = float(a.qr_arclen[nxt])
                a.request_frame = None
                a.waiting_for = ""
                a.actual_speed = a.commanded_speed
            else:
                # DENY: hold position, keep queue place (request_frame).
                a.waiting_for = blocker.name
                a.actual_speed = 0.0

    def _cell_blocker(
        self,
        requester: AMR,
        target: np.ndarray,
        active: list[AMR],
        disabled: list[AMR],
    ) -> AMR | None:
        """Return the AMR blocking ``target``, or None if the cell is free."""
        d = self.amr_safety_dist
        for b in active:
            if b is requester:
                continue
            near = (
                float(np.linalg.norm(b.position_at(b.progress) - target)) < d
                or float(np.linalg.norm(b.position_at(b.granted_s) - target)) < d
            )
            # Only a real blocker if the contested cell sits on b's line of
            # travel (rear-end or crossing). A peer passing alongside on a
            # parallel rail (cell offset to its side) does not block.
            if near and self._on_travel_line(b, target):
                return b
        # Disabled AMRs are static obstacles -- they block by pure proximity.
        for b in disabled:
            if float(np.linalg.norm(b.collision_pos - target)) < d:
                return b
        return None

    def _on_travel_line(self, b: AMR, target: np.ndarray) -> bool:
        """True if ``target`` is roughly along b's direction of travel (i.e. b
        will occupy it), rather than offset to b's side on a parallel lane."""
        bp = b.position_at(b.progress)
        h = b.heading_at(b.progress)
        to_t = np.asarray(target, dtype=float) - bp
        # Lateral offset = component of (target - b) perpendicular to b heading.
        lateral = abs(to_t[0] * math.sin(h) - to_t[1] * math.cos(h))
        return lateral < self.lateral_block_thresh


# ---------------------------------------------------------------------------
# AMR-Human collision check (current positions)
# ---------------------------------------------------------------------------


def amr_human_collision(
    amr: AMR,
    frame: int,
    workers: list[dict],
    collision_dist: float = 0.55,
) -> tuple[bool, str]:
    if not amr.is_active(frame):
        return False, ""
    pos = amr.position_at(amr.progress)
    for w in workers:
        if frame >= len(w["truth"]):
            continue
        wp = w["truth"][frame]
        if float(np.linalg.norm(pos - wp)) < collision_dist:
            return True, w["name"]
    return False, ""


# ---------------------------------------------------------------------------
# Conflict checker (space-time, AMR vs worker tubes)
# ---------------------------------------------------------------------------


# Step C now uses Step B's polygon helpers (point_in_polygon /
# polygon_signed_radius) directly. The old elliptic-distance helper is gone.


@dataclass
class ConflictResult:
    name: str
    status: str                        # CLEAR / WATCH / SLOWDOWN / REPLAN
    hard_mask: np.ndarray
    soft_mask: np.ndarray
    pred_positions: np.ndarray
    ttc: float                         # earliest hard-hit time (inf if none)
    closest_worker: str
    closest_worker_pos: np.ndarray | None
    margin: float                      # min normalized distance over horizon
    # AMR-speed gating: a stationary AMR cannot benefit from replanning, so
    # both the REPLAN window and the SLOWDOWN action are suppressed when
    # ``alpha_amr`` is near zero. ``t_replan_eff`` is the effective REPLAN
    # window used by both the status decision and the gantt cell coloring.
    alpha_amr: float = 1.0
    t_replan_eff: float = T_REPLAN_DEFAULT


class ConflictChecker:
    """Per-AMR space-time conflict check against tracked-worker safety tubes.

    Status decision (priority order; downstream planner action in brackets):

        REPLAN    hard hit AND ttc < t_replan_eff      [trigger Step D]
        SLOWDOWN  hard hit AND ttc >= t_replan_eff     [smooth decel]
        WATCH     only soft hit, OR alpha_amr ~= 0     [monitor]
        CLEAR     no hit                               [continue]

    where ``t_replan_eff = t_replan_base * alpha_amr(v_actual)`` and
    ``alpha_amr`` is a smoothstep from 0 (stationary) to 1 (at full speed).
    A stationary AMR can therefore never escalate to REPLAN: it has nothing
    to replan because it is not moving anywhere.
    """

    @staticmethod
    def check(
        amr: AMR,
        dt: float,
        T: int,
        worker_data: list[dict],
        t_replan: float = T_REPLAN_DEFAULT,
        v_amr_typical: float = V_AMR_TYPICAL_DEFAULT,
    ) -> ConflictResult:
        pred = amr.predicted_positions(dt, T)
        hard_mask = np.zeros(T, dtype=bool)
        soft_mask = np.zeros(T, dtype=bool)
        per_time_min_soft = np.full(T, np.inf)
        per_time_worst_worker = [""] * T
        per_time_worst_pos: list[np.ndarray | None] = [None] * T

        # Step B now produces asymmetric teardrop polygons rather than
        # axis-aligned ellipses; conflict checking is therefore a
        # point-in-polygon test rather than a normalised distance.
        for w in worker_data:
            for t in range(T):
                hard_poly = w["inflated"][t]
                soft_poly = w["soft"][t]
                in_hard = point_in_polygon(pred[t], hard_poly)
                in_soft = point_in_polygon(pred[t], soft_poly)
                hard_mask[t] |= in_hard
                soft_mask[t] |= in_soft
                # Cheap "how close to the soft tube" metric (1.0 at boundary,
                # <1 inside, >1 outside) used only for the cosmetic margin
                # field on the status card.
                d_soft = polygon_signed_radius(pred[t], soft_poly,
                                               centroid=w["centers"][t])
                if d_soft < per_time_min_soft[t]:
                    per_time_min_soft[t] = d_soft
                    per_time_worst_worker[t] = w["name"]
                    per_time_worst_pos[t] = w["centers"][t]

        # Velocity-gated REPLAN window. Reuse Step B's smoothstep gain so the
        # whole pipeline shares one "kinematic urgency" curve.
        alpha_amr = float(SafetyInflationModel.velocity_gain(
            float(amr.actual_speed), float(v_amr_typical)))
        t_replan_eff = float(t_replan * alpha_amr)
        # Below this gain the AMR is effectively at rest: REPLAN / SLOWDOWN
        # are not actionable, so any hit collapses to WATCH.
        alpha_dead = 0.05

        if hard_mask.any():
            t_star = int(np.argmax(hard_mask))
            ttc = (t_star + 1) * dt
            if alpha_amr < alpha_dead:
                status = "WATCH"
            elif ttc < t_replan_eff:
                status = "REPLAN"
            else:
                status = "SLOWDOWN"
            worker_name = per_time_worst_worker[t_star]
            worker_pos = per_time_worst_pos[t_star]
        elif soft_mask.any():
            status = "WATCH"
            t_star = int(np.argmax(soft_mask))
            ttc = float("inf")
            worker_name = per_time_worst_worker[t_star]
            worker_pos = per_time_worst_pos[t_star]
        else:
            status = "CLEAR"
            ttc = float("inf")
            worker_name = ""
            worker_pos = None

        return ConflictResult(
            name=amr.name, status=status,
            hard_mask=hard_mask, soft_mask=soft_mask,
            pred_positions=pred, ttc=ttc,
            closest_worker=worker_name, closest_worker_pos=worker_pos,
            margin=float(per_time_min_soft.min()),
            alpha_amr=alpha_amr,
            t_replan_eff=t_replan_eff,
        )
