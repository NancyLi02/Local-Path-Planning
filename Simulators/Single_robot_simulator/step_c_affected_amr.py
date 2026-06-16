"""
Step C: Affected AMR Identification (space-time conflict check) + simple
centralized AMR-AMR coordination layer.

Fleet model (matches the QR-code-on-the-floor factory abstraction)
------------------------------------------------------------------
- The floor is densely covered with QR codes (~1 m apart). Each AMR's
  reference path is therefore a chain of MANY intermediate QR waypoints,
  not just a start and a goal.
- Movement protocol: an AMR is only ever cleared to drive to its next
  granted QR waypoint. On arrival it reports to the ``CentralizedPlanner``
  and requests the next waypoint; until granted it physically waits on the
  QR cell it is parked on.
- AMR-AMR collision avoidance happens at the granting step: a waypoint is
  granted only if no other AMR occupies or has reserved a cell within
  ``amr_safety_dist`` of it. Competing requests are served first-come-
  first-served by **waypoint-arrival time** (the frame the AMR reached its
  current QR and filed the request) -- NOT by when the AMR spawned.
- Disabled (collided) AMRs keep occupying their cell, so traffic behind
  them queues up until a human supervisor would clear the wreck.
- AMRs are **spawned at staggered frame numbers** (not all together) and
  **traverse the path exactly once** (no looping).

Per-frame status hierarchy (most severe wins for display)
---------------------------------------------------------
The hierarchy is reorganised around the *downstream planner action* each
status implies, with TTC (time-to-conflict) deciding when "do something"
becomes "REPLAN NOW":

- PENDING        gray         AMR has not yet spawned
- COLLISION      charcoal     Physical contact already happened. *Sticky*:
                              the AMR is frozen at the collision point and
                              never moves again; its gantt row is frozen too.
- WAITING        indigo       CentralizedPlanner denied the next QR cell
                              (held/reserved by another AMR); the AMR parks
                              on its current QR waypoint until granted
- REPLAN         red          Hard-tube hit at TTC < ``t_replan_eff`` (~2.8 s
                              at full speed). Trigger Step D local replanning.
- SLOWDOWN       amber        Hard-tube hit at TTC >= ``t_replan_eff``. Smooth
                              deceleration is enough -- no replan needed.
- WATCH          lime         Only the soft tube is touched, OR the AMR is
                              essentially stationary so no replan/slowdown is
                              actionable. Keep watching, do not change path.
- CLEAR          green        No conflict at all in the 5 s horizon
- DONE           light gray   AMR has reached the end of its reference path

The effective threshold scales with the AMR's *current* speed:

    t_replan_eff = t_replan_base * alpha_amr(v_actual)
    alpha_amr    = smoothstep(v_actual / v_amr_typical)

so a fully stopped AMR (``v_actual = 0``) has ``t_replan_eff = 0`` and can
never escalate to REPLAN: there is nothing to replan when the platform is
already at rest. A slow-moving AMR has a correspondingly tight REPLAN window.
This matches Step B's worker-speed gating philosophy.

Sticky-collision freeze
-----------------------
When an AMR physically collides with any human it is marked ``collided``
permanently. From that frame onward:
  * its body, halo, ghosts, and BOOM marker stay pinned at the collision
    pose; the human keeps walking on top of it,
  * its gantt row is **frozen** at the hard/soft conflict pattern that was
    visible at the collision instant -- it does not advance any more, so the
    user sees a clean post-mortem of the conflict that caused the failure.

Visualization
-------------
Independent figure (kept separate from Step A/B):
  * Main map: 6 AMR rails (color-tagged at the start QR) + worker safety tubes
    + spawned AMR bodies with status halos + 5 future ghosts per AMR
    (FILL = AMR identity color, BORDER = status color at that ghost's t) +
    REPLAN connector (red) only on AMRs whose TTC < t_replan + explosion
    stars on physical AMR-human COLLISION.
  * Right column: 6 status cards (badge + speed + TTC + reason).
  * Bottom: per-AMR space-time Gantt (CLEAR / WATCH / SLOWDOWN / REPLAN
    cells) with a vertical t_replan threshold line.

Run
---
    python step_c_affected_amr.py
    python step_c_affected_amr.py --preview
    python step_c_affected_amr.py --frames 180 --amrs 6
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon, Rectangle

try:
    from .human_predict import (
        GOALS, MAP_BOUNDS, OBSTACLES,
        IntentParticlePredictor, PredictorConfig,
        _draw_factory, make_workers,
    )
    from .safety_inflation import (
        SafetyInflationConfig, SafetyInflationModel,
        safety_tube_polygon, point_in_polygon, polygon_signed_radius,
    )
except ImportError:                                # pragma: no cover
    from human_predict import (
        GOALS, MAP_BOUNDS, OBSTACLES,
        IntentParticlePredictor, PredictorConfig,
        _draw_factory, make_workers,
    )
    from safety_inflation import (
        SafetyInflationConfig, SafetyInflationModel,
        safety_tube_polygon, point_in_polygon, polygon_signed_radius,
    )


# ---------------------------------------------------------------------------
# Status palette
# ---------------------------------------------------------------------------

STATUS_COLOR: dict[str, str] = {
    "PENDING":   "#9e9e9e",
    "CLEAR":     "#2ca02c",
    "WATCH":     "#c0ca33",      # lime  -- soft hit, observe only
    "SLOWDOWN":  "#f9a825",      # amber -- hard hit, decelerate
    "REPLAN":    "#d62728",      # red   -- TTC < t_replan, trigger Step D
    "WAITING":   "#3949ab",
    "COLLISION": "#212121",      # charcoal -- chosen distinct from AMR-C purple
    "DONE":      "#bdbdbd",
}

# TTC threshold (at full AMR speed) above which graceful slowdown suffices.
# Scaled with the slower fleet speed so the spatial lookahead (~1.0 m at
# v_amr * t_replan) stays comparable to the faster demo (0.65 m/s * 1.5 s).
T_REPLAN_DEFAULT: float = 2.8

# Nominal fleet speed -- every AMR uses this commanded speed.
AMR_SPEED_DEFAULT: float = 0.35          # m/s  (~1.3 km/h, cautious factory pace)

# Reference AMR speed for alpha_amr velocity gating (matches fleet nominal).
V_AMR_TYPICAL_DEFAULT: float = AMR_SPEED_DEFAULT


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    return (*mcolors.to_rgb(color), float(np.clip(alpha, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# AMR
# ---------------------------------------------------------------------------


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
    ):
        self.amr_safety_dist = float(amr_safety_dist)
        self.dt = dt

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
            # Physical occupancy of the target cell.
            if float(np.linalg.norm(b.position_at(b.progress) - target)) < d:
                return b
            # Cell already reserved: b was granted a waypoint near target.
            if float(np.linalg.norm(b.position_at(b.granted_s) - target)) < d:
                return b
        for b in disabled:
            if float(np.linalg.norm(b.collision_pos - target)) < d:
                return b
        return None


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


# ---------------------------------------------------------------------------
# Fleet factory  (6 AMRs, staggered spawn, slower speeds, no looping)
# ---------------------------------------------------------------------------


def make_stray_loader(
    num_frames: int,
    dt: float,
    collision_frame: int = 93,
    collision_xy: tuple[float, float] = (10.0, 6.5),
    walk_speed: float = 1.0,
) -> dict:
    """Untracked 'Loader' walking through the central aisle.

    The Loader is NOT seen by Step A's intent predictor and NOT wrapped in a
    Step B safety tube. From the AMR's perspective they are an unmodeled
    intruder. Their truth path is engineered to bring them through
    ``collision_xy`` at exactly ``collision_frame`` -- the time AMR-B passes
    that same point -- so the demo will reliably exhibit the physical
    AMR-human collision that the rest of the system is designed to handle.
    """
    cx, cy = collision_xy
    start = np.array([cx, cy + 3.0])                  # 3 m above the lane
    end = np.array([cx, cy - 4.5])                    # walks straight south
    dist_to_target = 3.0
    frames_to_target = int(round(dist_to_target / walk_speed / dt))
    spawn_frame = max(0, collision_frame - frames_to_target)

    truth = np.zeros((num_frames, 2))
    seg_len = float(np.linalg.norm(end - start))
    direction = (end - start) / max(seg_len, 1e-9)
    for f in range(num_frames):
        if f < spawn_frame:
            truth[f] = start
        else:
            s = walk_speed * (f - spawn_frame) * dt
            s = min(s, seg_len)
            truth[f] = start + direction * s
    return dict(
        name="Loader",
        color="#d84315",                              # burnt orange
        truth=truth,
        spawn_frame=spawn_frame,
        is_tracked=False,
    )


def make_amrs(num_amrs: int = 6) -> list[AMR]:
    """Staggered fleet with two engineered AMR-AMR crossings to showcase yielding.

    Engineered conflicts (resolved by QR-cell reservation, FCFS by
    waypoint-arrival time):
        * AMR-A (east y=5) vs AMR-D (north x=7)    -> contest cell near (7, 5)
        * AMR-A (east y=5) vs AMR-C (south x=12.5) -> contest cell near (12.5, 5)
    Whoever files the request for the contested QR cell first wins; the other
    AMR holds at its current QR waypoint (WAITING) until the cell clears.
    """
    fleet_spec = [
        # name,    color,        waypoints,                            spawn
        ("AMR-A", "#0aa6a6",
         np.array([[0.5, 5.0], [19.5, 5.0]]),                              0),
        ("AMR-B", "#2d3e8c",
         np.array([[19.5, 6.5], [0.5, 6.5]]),                               12),
        ("AMR-D", "#5d6d00",
         np.array([[7.0, 3.5], [7.0, 10.5]]),                               30),
        ("AMR-C", "#7b1fa2",
         np.array([[12.5, 10.5], [12.5, 3.5]]),                             35),
        ("AMR-E", "#c2185b",
         np.array([[0.5, 7.5], [19.5, 7.5]]),                               55),
        ("AMR-F", "#00897b",
         np.array([[18.5, 3.5], [18.5, 10.5]]),                             70),
    ]
    fleet = [
        AMR(name=n, color=c, waypoints=wp,
            commanded_speed=AMR_SPEED_DEFAULT, spawn_frame=sf)
        for (n, c, wp, sf) in fleet_spec[:num_amrs]
    ]
    # Display order = alphabetical by name (A, B, C, D, E, F).
    # AMR-AMR arbitration order is decided per-request inside
    # CentralizedPlanner by waypoint-arrival time, not by this list order.
    fleet.sort(key=lambda a: a.name)
    return fleet


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _amr_body_polygon(
    cx: float, cy: float, heading: float,
    length: float = 1.05, width: float = 0.62,
) -> np.ndarray:
    half_l, half_w = length / 2, width / 2
    tip = half_l + 0.22
    local = np.array([
        [-half_l, -half_w],
        [+half_l * 0.65, -half_w],
        [tip, 0.0],
        [+half_l * 0.65, +half_w],
        [-half_l, +half_w],
    ])
    ca, sa = math.cos(heading), math.sin(heading)
    rot = np.array([[ca, -sa], [sa, ca]])
    return local @ rot.T + np.array([cx, cy])


def _amr_tag_position(cx: float, cy: float, heading: float) -> tuple[float, float]:
    """Place the name badge just behind the AMR body, offset to the left."""
    # Rear of body + slight lateral offset so the label clears the nose.
    tx = cx - 0.50 * math.cos(heading) - 0.30 * math.sin(heading)
    ty = cy - 0.50 * math.sin(heading) + 0.30 * math.cos(heading)
    return tx, ty


def _ghost_border_color(hard: bool, soft: bool, t_value: float,
                        t_replan_eff: float, alpha_amr: float = 1.0) -> str:
    """Per-ghost border color reflecting the planner action implied at that t.

    A near-stationary AMR (``alpha_amr ~= 0``) cannot replan or slow down, so
    every hit (hard or soft) collapses to WATCH for consistency with the
    overall status and gantt cells.
    """
    if alpha_amr < 0.05:
        return STATUS_COLOR["WATCH"] if (hard or soft) else STATUS_COLOR["CLEAR"]
    if hard:
        return STATUS_COLOR["REPLAN"] if t_value < t_replan_eff else STATUS_COLOR["SLOWDOWN"]
    if soft:
        return STATUS_COLOR["WATCH"]
    return STATUS_COLOR["CLEAR"]


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def build_step_c_animation(
    output_path: Path | None,
    num_frames: int = 280,
    num_workers: int = 2,
    num_amrs: int = 6,
    preview: bool = False,
    seed: int = 7,
    ghost_times_s: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    amr_safety_dist: float = 1.10,
    human_collision_dist: float = 0.55,
    t_replan: float = T_REPLAN_DEFAULT,
    v_amr_typical: float = V_AMR_TYPICAL_DEFAULT,
    inject_stray: bool = False,
):
    cfg = PredictorConfig(seed=seed)
    safety_cfg = SafetyInflationConfig()
    safety = SafetyInflationModel(safety_cfg)

    workers = make_workers(num_frames, cfg.dt, num_workers)
    predictors = [
        IntentParticlePredictor(
            GOALS, OBSTACLES, cfg,
            rng=np.random.default_rng(seed + 17 * i),
        )
        for i in range(len(workers))
    ]
    # Untracked humans -- no predictor, no safety tube. Collision check still
    # picks them up via direct geometric overlap.
    strays: list[dict] = []
    if inject_stray:
        strays.append(make_stray_loader(num_frames, cfg.dt))
    amrs = make_amrs(num_amrs)
    planner = CentralizedPlanner(
        amr_safety_dist=amr_safety_dist, dt=cfg.dt,
    )

    horizon_T = cfg.horizon_steps
    dt = cfg.dt
    horizon_seconds = horizon_T * dt
    ghost_idx = np.clip(
        np.array([int(round(t / dt)) - 1 for t in ghost_times_s]),
        0, horizon_T - 1,
    )

    # ----- figure layout -----
    fig = plt.figure(figsize=(16.0, 10.5), dpi=120)
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[3.0, 1.10],
        height_ratios=[11.5, 4.5, 0.7],
        wspace=0.18, hspace=0.32,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_status = fig.add_subplot(gs[0, 1])
    ax_gantt = fig.add_subplot(gs[1, :])
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    ax_status.axis("off")
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)

    _draw_factory(ax_map)
    ax_map.set_title(
        "Step C: Affected-AMR identification + centralized AMR-AMR yielding",
        fontsize=13, pad=10,
    )

    # ----- AMR rails (faint background lines) + QR markers -----
    for amr in amrs:
        ax_map.plot(amr.waypoints[:, 0], amr.waypoints[:, 1],
                    "-", color="#bdbdbd", lw=6.5, alpha=0.40,
                    solid_capstyle="round", zorder=1)
        ax_map.plot(amr.waypoints[:, 0], amr.waypoints[:, 1],
                    "--", color=amr.color, lw=1.1, alpha=0.55, zorder=1)
        # dense intermediate QR codes (~1 m apart) along the rail
        if len(amr.qr_points) > 2:
            ax_map.scatter(amr.qr_points[1:-1, 0], amr.qr_points[1:-1, 1],
                           marker="D", s=12, facecolor="white",
                           edgecolor="#8d8d8d", lw=0.6, alpha=0.85, zorder=2)
        # start QR marker
        p_start = amr.waypoints[0]
        ax_map.scatter(p_start[0], p_start[1], marker="s", s=44,
                       facecolor="white", edgecolor=amr.color, lw=1.6, zorder=2)
        # end QR marker
        p_end = amr.waypoints[-1]
        ax_map.scatter(p_end[0], p_end[1], marker="o", s=44,
                       facecolor=amr.color, edgecolor="white", lw=1.4, zorder=2)

    # ----- tracked worker handles (Step A predicted, Step B inflated) -----
    placeholder_poly = np.array([[0.0, 0.0], [0.0, 1e-3], [1e-3, 0.0]])
    worker_artists = []
    for w in workers:
        tube = Polygon(placeholder_poly, closed=True,
                       facecolor=_rgba(w["color"], 0.09),
                       edgecolor=_rgba(w["color"], 0.45),
                       lw=1.0, zorder=2)
        ax_map.add_patch(tube)
        hist_line, = ax_map.plot([], [], "-", lw=1.4,
                                 color=_rgba(w["color"], 0.50), zorder=3)
        dot = Circle((0, 0), 0.22, facecolor=w["color"],
                     edgecolor="white", lw=1.0, zorder=6)
        ax_map.add_patch(dot)
        tag = ax_map.text(
            0, 0, w["name"], fontsize=8.5, color="white",
            fontweight="bold", ha="center", va="bottom",
            bbox=dict(facecolor=w["color"], edgecolor="white",
                      lw=0.6, pad=2, boxstyle="round,pad=0.18"),
            zorder=7,
        )
        worker_artists.append(dict(tube=tube, hist=hist_line, dot=dot, tag=tag))

    # ----- stray (untracked) worker handles -- no tube, just a marked body
    stray_artists = []
    for s in strays:
        # Draw the full path lightly so the user sees the intruder ahead of time.
        ax_map.plot(s["truth"][:, 0], s["truth"][:, 1], ":",
                    lw=1.0, color=_rgba(s["color"], 0.45), zorder=1)
        hist_line, = ax_map.plot([], [], "-", lw=1.4,
                                 color=_rgba(s["color"], 0.55), zorder=3)
        dot = Circle((0, 0), 0.22, facecolor=s["color"],
                     edgecolor="white", lw=1.4, zorder=6,
                     hatch="//", alpha=0.0)
        ax_map.add_patch(dot)
        tag = ax_map.text(0, 0, f"{s['name']} (untracked)",
                          fontsize=7.5, color="white", fontweight="bold",
                          ha="center", va="bottom",
                          bbox=dict(facecolor=s["color"], edgecolor="white",
                                    lw=0.5, pad=1.4, boxstyle="round,pad=0.18"),
                          alpha=0.0, zorder=7)
        stray_artists.append(dict(hist=hist_line, dot=dot, tag=tag))

    # ----- AMR handles -----
    amr_artists = []
    for amr in amrs:
        body = Polygon(_amr_body_polygon(amr.waypoints[0, 0], amr.waypoints[0, 1], 0.0),
                       closed=True,
                       facecolor=amr.color, edgecolor="white", lw=1.5,
                       alpha=0.0, zorder=6)
        ax_map.add_patch(body)
        halo = Circle((0, 0), 0.70, facecolor="none",
                      edgecolor=STATUS_COLOR["CLEAR"], lw=2.5,
                      alpha=0.0, zorder=5)
        ax_map.add_patch(halo)
        planned_line, = ax_map.plot([], [], "-", lw=1.6,
                                    color=_rgba(amr.color, 0.85), zorder=4)
        ghosts = []
        for _ in ghost_times_s:
            g = Circle((0, 0), 0.20,
                       facecolor=amr.color,
                       edgecolor=STATUS_COLOR["CLEAR"], lw=1.8,
                       alpha=0.0, zorder=4)
            ax_map.add_patch(g)
            ghosts.append(g)
        conflict_line, = ax_map.plot([], [], "-", lw=1.6,
                                     color=STATUS_COLOR["REPLAN"],
                                     alpha=0.0, zorder=5)
        conflict_marker = Circle((0, 0), 0.30, facecolor="none",
                                 edgecolor=STATUS_COLOR["REPLAN"], lw=2.0,
                                 alpha=0.0, zorder=5)
        ax_map.add_patch(conflict_marker)
        # Collision explosion star (yellow filled, red outline)
        boom, = ax_map.plot([], [], marker="*", markersize=32,
                            markerfacecolor="#fff176",
                            markeredgecolor=STATUS_COLOR["COLLISION"],
                            markeredgewidth=2.2, linestyle="None",
                            alpha=0.0, zorder=7)
        boom_text = ax_map.text(0, 0, "BOOM", fontsize=8,
                                color=STATUS_COLOR["COLLISION"],
                                fontweight="bold", ha="center", va="bottom",
                                alpha=0.0, zorder=7)
        tag = ax_map.text(
            0, 0, amr.name, fontsize=8.5, color="white",
            fontweight="bold", ha="center", va="center",
            bbox=dict(facecolor=amr.color, edgecolor="white",
                      lw=0.6, pad=2, boxstyle="round,pad=0.18"),
            alpha=0.0, zorder=8,
        )
        amr_artists.append(dict(
            body=body, halo=halo, planned=planned_line,
            ghosts=ghosts, conf_line=conflict_line, conf_mark=conflict_marker,
            boom=boom, boom_text=boom_text, tag=tag,
        ))

    info_text = ax_map.text(
        MAP_BOUNDS[0] + 0.4, MAP_BOUNDS[3] - 0.3, "",
        fontsize=9, color="#222",
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        zorder=8, va="top", ha="left",
    )

    # ----- right-side per-AMR status cards (6 compact cards) -----
    n_cards = len(amrs)
    ax_status.text(
        0.5, 0.995, "Fleet status",
        fontsize=11.5, fontweight="bold", color="#222",
        ha="center", va="top", transform=ax_status.transAxes,
    )
    top_margin = 0.045
    bot_margin = 0.020
    gap = 0.013
    card_height = (1.0 - top_margin - bot_margin - gap * (n_cards - 1)) / n_cards
    card_handles = []
    for i, amr in enumerate(amrs):
        top = 1.0 - top_margin - i * (card_height + gap)
        bot = top - card_height
        bg = FancyBboxPatch(
            (0.02, bot + 0.003), 0.96, card_height - 0.006,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor="white", edgecolor=amr.color, lw=1.5,
            transform=ax_status.transAxes,
        )
        ax_status.add_patch(bg)
        # identity dot + name
        y_name = top - card_height * 0.22
        ax_status.text(0.05, y_name, "\u25cf",
                       fontsize=14, color=amr.color,
                       transform=ax_status.transAxes, va="center")
        name_txt = ax_status.text(0.10, y_name, amr.name,
                                  fontsize=10.5, fontweight="bold", color="#222",
                                  transform=ax_status.transAxes, va="center")
        speed_txt = ax_status.text(
            0.97, y_name, f"v={amr.commanded_speed:.2f}",
            fontsize=8.5, color="#444",
            transform=ax_status.transAxes, va="center", ha="right",
        )
        # status badge
        y_badge = top - card_height * 0.56
        badge_w = 0.30
        badge_x = 0.05
        badge = FancyBboxPatch(
            (badge_x, y_badge - card_height * 0.17),
            badge_w, card_height * 0.32,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            facecolor="#cccccc", edgecolor="none",
            transform=ax_status.transAxes,
        )
        ax_status.add_patch(badge)
        badge_text = ax_status.text(
            badge_x + badge_w * 0.5, y_badge, "—",
            fontsize=9.0, color="white", fontweight="bold",
            ha="center", va="center",
            transform=ax_status.transAxes,
        )
        # detail line (right of badge)
        detail_text = ax_status.text(
            badge_x + badge_w + 0.025, y_badge, "",
            fontsize=8.7, color="#222",
            transform=ax_status.transAxes, va="center",
        )
        # secondary line (below badge) for the cause / partner
        y_sub = top - card_height * 0.86
        sub_text = ax_status.text(
            0.06, y_sub, "",
            fontsize=8.0, color="#555",
            transform=ax_status.transAxes, va="center",
        )
        card_handles.append(dict(
            badge=badge, badge_text=badge_text,
            detail=detail_text, sub=sub_text,
            name=name_txt, speed=speed_txt,
            amr=amr,
        ))

    # ----- bottom Gantt chart -----
    ax_gantt.set_xlim(0, horizon_seconds)
    ax_gantt.set_ylim(-0.5, n_cards - 0.5)
    ax_gantt.invert_yaxis()
    ax_gantt.set_yticks(range(n_cards))
    ax_gantt.set_yticklabels([amr.name for amr in amrs], fontsize=9)
    ax_gantt.set_xlabel("Lookahead horizon t [s]")
    ax_gantt.set_title(
        f"Per-AMR space-time conflict status "
        f"(red zone < t_replan_eff = α_amr(v)·{t_replan:.1f}s  →  trigger Step D; "
        "α_amr=0 ⇒ stationary AMR never REPLANs)",
        fontsize=10.5, pad=6,
    )
    ax_gantt.set_facecolor("#fafafa")
    ax_gantt.grid(True, alpha=0.30, axis="x")
    for spine in ("top", "right"):
        ax_gantt.spines[spine].set_visible(False)
    ax_gantt.axvline(0.0, color="#444", lw=1.0)
    ax_gantt.axvline(horizon_seconds, color="#444", lw=1.0, linestyle="--", alpha=0.7)
    # REPLAN threshold line: any hard cell to its LEFT promotes the AMR to REPLAN.
    ax_gantt.axvline(t_replan, color=STATUS_COLOR["REPLAN"], lw=1.8, linestyle="--",
                     alpha=0.8)
    ax_gantt.text(
        t_replan - 0.05, -0.55, f"t_replan = {t_replan:.1f}s",
        fontsize=8.5, color=STATUS_COLOR["REPLAN"], ha="right", va="top",
        fontweight="bold",
    )

    gantt_cells = []
    gantt_collision_stars = []          # black star at t=0 of a collided row
    gantt_collision_labels = []         # right-side text annotation
    gantt_collision_overlays = []       # black hatched overlay across the row
    for ai in range(n_cards):
        row_cells = []
        for t in range(horizon_T):
            rect = Rectangle((t * dt, ai - 0.36), dt, 0.72,
                             facecolor=STATUS_COLOR["PENDING"],
                             edgecolor="white", lw=0.4)
            ax_gantt.add_patch(rect)
            row_cells.append(rect)
        gantt_cells.append(row_cells)

        # Diagonal black hatch overlay — drawn on top of the frozen pattern so
        # the user can still read the underlying conflict cells but immediately
        # recognises the row as a collided AMR. Hidden until collision.
        overlay = Rectangle(
            (0.0, ai - 0.36), horizon_seconds, 0.72,
            facecolor="none",
            edgecolor=STATUS_COLOR["COLLISION"],
            hatch="//", linewidth=0.0,
            alpha=0.0, zorder=4,
        )
        ax_gantt.add_patch(overlay)
        gantt_collision_overlays.append(overlay)

        # Big black star pinned at t=0 of the row.
        star, = ax_gantt.plot(
            [], [], marker="*", markersize=18,
            markerfacecolor=STATUS_COLOR["COLLISION"],
            markeredgecolor="white", markeredgewidth=1.4,
            linestyle="None", zorder=6, alpha=0.0,
        )
        gantt_collision_stars.append(star)

        # Right-side label "COLLISION @ frame N (Tech-1)".
        label = ax_gantt.text(
            horizon_seconds * 0.97, ai, "",
            fontsize=8.5, color="white", fontweight="bold",
            ha="right", va="center", alpha=0.0, zorder=7,
            bbox=dict(facecolor=STATUS_COLOR["COLLISION"],
                      edgecolor="white", linewidth=1.0,
                      boxstyle="round,pad=0.22"),
        )
        gantt_collision_labels.append(label)

    # ----- legend -----
    legend_handles = [
        plt.Line2D([0], [0], color="#bdbdbd", lw=6.5, alpha=0.65),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor="white", markeredgecolor="#888",
                   markersize=8, linestyle="None"),
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="white", markeredgecolor="#8d8d8d",
                   markersize=6, linestyle="None"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#888", markeredgecolor="white",
                   markersize=9, linestyle="None"),
        Patch(facecolor=_rgba("#888888", 0.10), edgecolor="#888"),
        Patch(facecolor="#888", edgecolor=STATUS_COLOR["CLEAR"], linewidth=2.0),
        Patch(facecolor="#888", edgecolor=STATUS_COLOR["WATCH"], linewidth=2.0),
        Patch(facecolor="#888", edgecolor=STATUS_COLOR["SLOWDOWN"], linewidth=2.0),
        Patch(facecolor="#888", edgecolor=STATUS_COLOR["REPLAN"], linewidth=2.0),
        plt.Line2D([0], [0], color=STATUS_COLOR["REPLAN"], lw=1.8),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#fff176",
                   markeredgecolor=STATUS_COLOR["COLLISION"],
                   markersize=18, linestyle="None"),
        Patch(facecolor="white", edgecolor=STATUS_COLOR["COLLISION"],
              hatch="//", linewidth=1.0),
    ]
    legend_labels = [
        "AMR reference rail",
        "QR start",
        "QR waypoint (dense floor grid)",
        "QR goal",
        "Tracked worker safety tube (Step B)",
        "Ghost border: CLEAR",
        "Ghost border: WATCH (soft hit, or stationary AMR)",
        "Ghost border: SLOWDOWN (hard, TTC ≥ t_replan_eff)",
        "Ghost border: REPLAN (hard, TTC < t_replan_eff)",
        "REPLAN connector (ghost @ TTC → worker)",
        "Physical AMR-human collision (AMR disabled)",
        "Gantt row: collided AMR (hatched overlay + ★)",
    ]
    if strays:
        # When a stray is injected, surface it explicitly in the legend.
        legend_handles.insert(4,
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#d84315", markeredgecolor="white",
                       markersize=10, linestyle="None"))
        legend_labels.insert(4, "Untracked human (Loader, no tube)")
    ax_leg.legend(legend_handles, legend_labels, loc="center", ncol=4,
                  frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=8.0)

    # ----- per-frame update -----
    def update(frame: int):
        # 1a) Tracked workers + Step-A predictions + Step-B teardrop lobes
        worker_data = []
        for w, predictor, art in zip(workers, predictors, worker_artists):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            out = predictor.rollout(obs)
            ellipses = out["ellipses"]
            # Pass mean_traj so the buffer auto-shrinks when the worker
            # stops (alpha_h gating) AND points forward when they walk
            # (heading -> teardrop orientation).
            hard_lobes, soft_lobes, _ = safety.inflate_all(
                ellipses, dt, out["belief"], mean_traj=out["mean"],
            )
            centers = ellipses[:, :2]

            hull = safety_tube_polygon(hard_lobes)
            art["tube"].set_xy(hull)
            art["hist"].set_data(w["truth"][: frame + 1, 0],
                                 w["truth"][: frame + 1, 1])
            art["dot"].center = tuple(w["truth"][frame])
            art["tag"].set_position((float(w["truth"][frame, 0]),
                                     float(w["truth"][frame, 1]) + 0.42))

            worker_data.append(dict(
                name=w["name"], color=w["color"],
                inflated=hard_lobes, soft=soft_lobes,
                centers=centers,
            ))

        # 1b) Stray (untracked) humans -- no prediction, no tube. Only their
        # ground-truth position participates, via the collision check.
        all_humans = list(workers)
        for s, sart in zip(strays, stray_artists):
            spawned = frame >= s.get("spawn_frame", 0)
            pos = s["truth"][frame] if frame < len(s["truth"]) else s["truth"][-1]
            sart["dot"].center = tuple(pos)
            sart["dot"].set_alpha(1.0 if spawned else 0.0)
            sart["hist"].set_data(s["truth"][: frame + 1, 0],
                                  s["truth"][: frame + 1, 1])
            sart["tag"].set_position((pos[0], pos[1] + 0.6))
            sart["tag"].set_alpha(1.0 if spawned else 0.0)
            all_humans.append(s)

        # 2) Centralized planner sets actual_speed for all active AMRs
        # (collided AMRs are filtered out via is_active and stay frozen)
        planner.resolve(amrs, frame)

        # 3) Per-AMR conflict + collision + status + render
        any_collision_now = False
        any_replan = 0
        any_slowdown = 0
        any_waiting = 0
        for ai, (amr, art, card) in enumerate(zip(amrs, amr_artists, card_handles)):
            spawned = amr.is_spawned(frame)
            done = amr.is_done()

            # ------------------------------------------------------------
            # Inactive (PENDING / DONE / COLLIDED-but-not-yet-rendered)
            # ------------------------------------------------------------
            if amr.collided:
                # Persistent frozen rendering at collision pose.
                cx, cy = amr.collision_pos
                head = amr.collision_heading
                art["body"].set_xy(_amr_body_polygon(cx, cy, head))
                art["body"].set_facecolor(amr.color)
                art["body"].set_alpha(0.45)            # desaturated to read "disabled"
                art["halo"].center = (cx, cy)
                art["halo"].set_edgecolor(STATUS_COLOR["COLLISION"])
                art["halo"].set_alpha(0.95)
                art["halo"].set_linewidth(3.2)
                art["planned"].set_data([], [])
                for g in art["ghosts"]:
                    g.set_alpha(0.0)
                art["conf_line"].set_alpha(0.0)
                art["conf_mark"].set_alpha(0.0)
                art["boom"].set_data([cx], [cy])
                art["boom"].set_alpha(1.0)
                art["boom_text"].set_position((cx, cy + 0.55))
                art["boom_text"].set_alpha(1.0)
                art["boom_text"].set_text("DISABLED")
                ltx, lty = _amr_tag_position(cx, cy, head)
                art["tag"].set_position((ltx, lty))
                art["tag"].set_alpha(0.85)

                # Gantt row is FROZEN at the conflict pattern captured at the
                # moment of collision -- it does not advance any more. We just
                # re-stamp it every frame so animation blitting can pick it up.
                hard_mask = amr.frozen_hard_mask
                soft_mask = amr.frozen_soft_mask
                t_replan_eff_f = amr.frozen_t_replan_eff
                alpha_amr_f = amr.frozen_alpha_amr
                for t in range(horizon_T):
                    t_val = (t + 1) * dt
                    if hard_mask is not None and bool(hard_mask[t]):
                        if alpha_amr_f < 0.05:
                            color = STATUS_COLOR["WATCH"]
                        elif t_val < t_replan_eff_f:
                            color = STATUS_COLOR["REPLAN"]
                        else:
                            color = STATUS_COLOR["SLOWDOWN"]
                    elif soft_mask is not None and bool(soft_mask[t]):
                        color = STATUS_COLOR["WATCH"]
                    else:
                        color = STATUS_COLOR["CLEAR"]
                    gantt_cells[ai][t].set_facecolor(color)

                # Persistent collision indicators on the gantt row.
                gantt_collision_overlays[ai].set_alpha(0.55)
                gantt_collision_stars[ai].set_data([0.06], [ai])
                gantt_collision_stars[ai].set_alpha(1.0)
                gantt_collision_labels[ai].set_text(
                    f"COLLISION @ f{amr.collision_frame}  ({amr.collision_worker})"
                )
                gantt_collision_labels[ai].set_alpha(1.0)

                card["badge"].set_facecolor(STATUS_COLOR["COLLISION"])
                card["badge_text"].set_text("COLLISION")
                card["detail"].set_text(f"hit {amr.collision_worker} @ frame {amr.collision_frame}")
                card["sub"].set_text("AMR frozen / gantt locked at impact")
                card["speed"].set_text("v=0.00")
                continue

            if not spawned or done:
                art["body"].set_alpha(0.0)
                art["halo"].set_alpha(0.0)
                art["planned"].set_data([], [])
                for g in art["ghosts"]:
                    g.set_alpha(0.0)
                art["conf_line"].set_alpha(0.0)
                art["conf_mark"].set_alpha(0.0)
                art["boom"].set_alpha(0.0)
                art["boom_text"].set_alpha(0.0)
                art["tag"].set_alpha(0.0)

                if not spawned:
                    status = "PENDING"
                    detail = f"spawns @ frame {amr.spawn_frame}"
                else:
                    status = "DONE"
                    detail = "path completed"

                for t in range(horizon_T):
                    gantt_cells[ai][t].set_facecolor(STATUS_COLOR[status])

                # No collision on this row.
                gantt_collision_overlays[ai].set_alpha(0.0)
                gantt_collision_stars[ai].set_alpha(0.0)
                gantt_collision_labels[ai].set_alpha(0.0)

                card["badge"].set_facecolor(STATUS_COLOR[status])
                card["badge_text"].set_text(status)
                card["detail"].set_text(detail)
                card["sub"].set_text("")
                card["speed"].set_text("v=—")
                continue

            # ------------------------------------------------------------
            # Active AMR: check for new physical collision FIRST
            # ------------------------------------------------------------
            collided_now, hit_worker = amr_human_collision(
                amr, frame, all_humans, collision_dist=human_collision_dist,
            )
            if collided_now:
                # Run the Step-C check once so we can SNAPSHOT the conflict
                # pattern at the instant of impact -- this becomes the frozen
                # gantt row for the rest of the simulation.
                impact_result = ConflictChecker.check(
                    amr, dt, horizon_T, worker_data,
                    t_replan=t_replan, v_amr_typical=v_amr_typical,
                )
                amr.mark_collision(
                    frame, hit_worker,
                    hard_mask=impact_result.hard_mask,
                    soft_mask=impact_result.soft_mask,
                    alpha_amr=impact_result.alpha_amr,
                    t_replan_eff=impact_result.t_replan_eff,
                )
                any_collision_now = True

                cx, cy = amr.collision_pos
                head = amr.collision_heading
                art["body"].set_xy(_amr_body_polygon(cx, cy, head))
                art["body"].set_facecolor(amr.color)
                art["body"].set_alpha(0.55)
                art["halo"].center = (cx, cy)
                art["halo"].set_edgecolor(STATUS_COLOR["COLLISION"])
                art["halo"].set_alpha(1.0)
                art["halo"].set_linewidth(3.5)
                art["planned"].set_data([], [])
                for g in art["ghosts"]:
                    g.set_alpha(0.0)
                art["conf_line"].set_alpha(0.0)
                art["conf_mark"].set_alpha(0.0)
                art["boom"].set_data([cx], [cy])
                art["boom"].set_alpha(1.0)
                art["boom_text"].set_position((cx, cy + 0.55))
                art["boom_text"].set_alpha(1.0)
                art["boom_text"].set_text(f"BOOM  ({hit_worker})")
                ltx, lty = _amr_tag_position(cx, cy, head)
                art["tag"].set_position((ltx, lty))
                art["tag"].set_alpha(0.85)

                # Lock in the gantt row at the impact-instant conflict pattern.
                alpha_amr_f = impact_result.alpha_amr
                t_replan_eff_f = impact_result.t_replan_eff
                for t in range(horizon_T):
                    t_val = (t + 1) * dt
                    if bool(impact_result.hard_mask[t]):
                        if alpha_amr_f < 0.05:
                            color = STATUS_COLOR["WATCH"]
                        elif t_val < t_replan_eff_f:
                            color = STATUS_COLOR["REPLAN"]
                        else:
                            color = STATUS_COLOR["SLOWDOWN"]
                    elif bool(impact_result.soft_mask[t]):
                        color = STATUS_COLOR["WATCH"]
                    else:
                        color = STATUS_COLOR["CLEAR"]
                    gantt_cells[ai][t].set_facecolor(color)

                # Persistent collision indicators on the gantt row.
                gantt_collision_overlays[ai].set_alpha(0.55)
                gantt_collision_stars[ai].set_data([0.06], [ai])
                gantt_collision_stars[ai].set_alpha(1.0)
                gantt_collision_labels[ai].set_text(
                    f"COLLISION @ f{amr.collision_frame}  ({amr.collision_worker})"
                )
                gantt_collision_labels[ai].set_alpha(1.0)

                card["badge"].set_facecolor(STATUS_COLOR["COLLISION"])
                card["badge_text"].set_text("COLLISION")
                card["detail"].set_text(f"hit  {hit_worker}")
                card["sub"].set_text("AMR frozen / gantt locked at impact")
                card["speed"].set_text("v=0.00")
                continue

            # Step-C check uses ACTUAL speed (WAITING -> stationary projection)
            # The checker also velocity-gates REPLAN against amr.actual_speed,
            # so a stationary AMR (alpha_amr ~= 0) can never escalate beyond
            # WATCH no matter how close the worker tube comes.
            result = ConflictChecker.check(
                amr, dt, horizon_T, worker_data,
                t_replan=t_replan, v_amr_typical=v_amr_typical,
            )

            # Status priority:  WAITING > REPLAN > SLOWDOWN > WATCH > CLEAR
            # (WAITING dominates because the centralised planner has already
            # acted; Step-C info appears in the sub-line.)
            t_replan_eff = result.t_replan_eff
            if amr.waiting_for:
                status = "WAITING"
                detail = (f"hold @ QR#{amr.current_qr_index()}  "
                          f"(cell held by {amr.waiting_for})")
                if result.status == "REPLAN":
                    sub = f"Step-C: REPLAN, {result.closest_worker} @ {result.ttc:.1f}s"
                elif result.status == "SLOWDOWN":
                    sub = f"Step-C: SLOWDOWN, {result.closest_worker} @ {result.ttc:.1f}s"
                elif result.status == "WATCH":
                    sub = f"Step-C: WATCH, near {result.closest_worker}"
                else:
                    sub = "Step-C: clear"
                any_waiting += 1
            elif result.status == "REPLAN":
                status = "REPLAN"
                detail = f"TTC = {result.ttc:.1f}s  <  {t_replan_eff:.2f}s"
                sub = f"trigger Step D  ({result.closest_worker})"
                any_replan += 1
            elif result.status == "SLOWDOWN":
                status = "SLOWDOWN"
                detail = f"TTC = {result.ttc:.1f}s"
                sub = f"decelerate, monitor  ({result.closest_worker})"
                any_slowdown += 1
            elif result.status == "WATCH":
                status = "WATCH"
                if result.hard_mask.any() and result.alpha_amr < 0.05:
                    detail = "stationary -- no replan"
                    sub = f"hold ground  ({result.closest_worker})"
                else:
                    detail = f"soft margin {result.margin:.2f}"
                    sub = f"observe  ({result.closest_worker})"
            else:
                status = "CLEAR"
                detail = f"margin {result.margin:.2f}"
                sub = "no conflict in 5 s"

            status_clr = STATUS_COLOR[status]

            # Body + halo
            cx, cy = amr.position_at(amr.progress)
            head = amr.heading_at(amr.progress)
            art["body"].set_xy(_amr_body_polygon(cx, cy, head))
            art["body"].set_facecolor(amr.color)
            art["body"].set_alpha(1.0)
            art["halo"].center = (cx, cy)
            art["halo"].set_edgecolor(status_clr)
            if status == "REPLAN":
                art["halo"].set_alpha(0.95)
                art["halo"].set_linewidth(3.0)
            elif status in ("WAITING", "SLOWDOWN"):
                art["halo"].set_alpha(0.80)
                art["halo"].set_linewidth(2.3)
            elif status == "WATCH":
                art["halo"].set_alpha(0.55)
                art["halo"].set_linewidth(1.8)
            else:
                art["halo"].set_alpha(0.20)
                art["halo"].set_linewidth(1.3)

            # Planned 5 s trajectory
            art["planned"].set_data(
                result.pred_positions[:, 0], result.pred_positions[:, 1],
            )

            # Future ghosts -- fill = AMR identity, border = per-t status color
            for k, t_idx in enumerate(ghost_idx):
                gpos = result.pred_positions[t_idx]
                t_val = (t_idx + 1) * dt
                g = art["ghosts"][k]
                g.center = (float(gpos[0]), float(gpos[1]))
                g.set_facecolor(_rgba(amr.color, 0.88 - 0.10 * k))
                g.set_edgecolor(_ghost_border_color(
                    bool(result.hard_mask[t_idx]),
                    bool(result.soft_mask[t_idx]),
                    t_val, t_replan_eff, result.alpha_amr,
                ))
                g.set_radius(max(0.23 - 0.025 * k, 0.10))
                g.set_alpha(0.92 - 0.08 * k)

            # REPLAN connector: only when the first hard-tube hit is inside t_replan
            if result.status == "REPLAN" and result.closest_worker_pos is not None:
                t_star = int(np.argmax(result.hard_mask))
                amr_at_t = result.pred_positions[t_star]
                wpos = result.closest_worker_pos
                art["conf_line"].set_data(
                    [amr_at_t[0], wpos[0]], [amr_at_t[1], wpos[1]],
                )
                art["conf_line"].set_alpha(0.90)
                art["conf_mark"].center = (float(amr_at_t[0]), float(amr_at_t[1]))
                art["conf_mark"].set_alpha(0.90)
            else:
                art["conf_line"].set_alpha(0.0)
                art["conf_mark"].set_alpha(0.0)

            # Boom is reserved for actual COLLISION rendering only
            art["boom"].set_alpha(0.0)
            art["boom_text"].set_alpha(0.0)

            # Name badge follows the AMR body.
            ltx, lty = _amr_tag_position(cx, cy, head)
            art["tag"].set_position((ltx, lty))
            art["tag"].set_alpha(1.0)

            # Card
            card["badge"].set_facecolor(status_clr)
            card["badge_text"].set_text(status)
            card["detail"].set_text(detail)
            card["sub"].set_text(sub)
            card["speed"].set_text(f"v={amr.actual_speed:.2f}")

            # Gantt row colors: per-cell time-slice severity. The cells use
            # the *effective* REPLAN window (which shrinks with AMR speed):
            # for a stationary AMR every hit collapses to WATCH because no
            # replanning is actionable.
            for t in range(horizon_T):
                t_val = (t + 1) * dt
                if result.hard_mask[t]:
                    if result.alpha_amr < 0.05:
                        color = STATUS_COLOR["WATCH"]
                    elif t_val < t_replan_eff:
                        color = STATUS_COLOR["REPLAN"]
                    else:
                        color = STATUS_COLOR["SLOWDOWN"]
                elif result.soft_mask[t]:
                    color = STATUS_COLOR["WATCH"]
                else:
                    color = STATUS_COLOR["CLEAR"]
                gantt_cells[ai][t].set_facecolor(color)

            # No collision on this row.
            gantt_collision_overlays[ai].set_alpha(0.0)
            gantt_collision_stars[ai].set_alpha(0.0)
            gantt_collision_labels[ai].set_alpha(0.0)

        # 4) Advance AMRs (after status snapshot for this frame is captured).
        # is_active already filters out collided / pending / done AMRs.
        for amr in amrs:
            if amr.is_active(frame):
                amr.step(dt)

        # Summary info
        n_active = sum(1 for a in amrs if a.is_active(frame))
        n_pending = sum(1 for a in amrs if not a.is_spawned(frame))
        n_done = sum(1 for a in amrs if a.is_done())
        n_collided = sum(1 for a in amrs if a.collided)
        info_text.set_text(
            f"frame {frame+1}/{num_frames}    "
            f"active {n_active} / pending {n_pending} / done {n_done} / collided {n_collided}    "
            f"REPLAN {any_replan}  SLOWDOWN {any_slowdown}  WAITING {any_waiting}"
            + ("    NEW COLLISION!" if any_collision_now else "")
            + f"\nCentralizedPlanner: QR-cell reservation, FCFS by waypoint-arrival   "
            f"Step-D trigger:  REPLAN  (TTC < t_replan = {t_replan:.1f}s)"
        )

        if not preview and (frame % 5 == 0 or frame == num_frames - 1):
            print(f"  rendering frame {frame+1}/{num_frames}")

        artists = [info_text]
        for art in worker_artists:
            artists += [art["tube"], art["hist"], art["dot"], art["tag"]]
        for sart in stray_artists:
            artists += [sart["hist"], sart["dot"], sart["tag"]]
        for art in amr_artists:
            artists += [art["body"], art["halo"], art["planned"],
                        art["conf_line"], art["conf_mark"],
                        art["boom"], art["boom_text"], art["tag"]]
            artists += art["ghosts"]
        for card in card_handles:
            artists += [card["badge"], card["badge_text"],
                        card["detail"], card["sub"], card["speed"]]
        for row in gantt_cells:
            artists += row
        artists += gantt_collision_overlays
        artists += gantt_collision_stars
        artists += gantt_collision_labels
        return artists

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

    if preview:
        print("Preview mode: showing live window. Close it to exit.")
        plt.show()
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(fps=10, bitrate=2400)
        anim.save(output_path, writer=writer)
        saved = output_path
    except Exception as exc:                                # pragma: no cover
        print(f"  ffmpeg unavailable ({exc}); falling back to GIF")
        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=8))
        saved = gif_path
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_output() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "step_c_affected_amr_demo.mp4"


def main(argv: list[str] | None = None) -> None:
    pa = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pa.add_argument("--output", type=str, default=str(_default_output()))
    pa.add_argument("--frames", type=int, default=280)
    pa.add_argument("--workers", type=int, default=2, choices=[1, 2, 3, 4])
    pa.add_argument("--amrs", type=int, default=6, choices=[1, 2, 3, 4, 5, 6])
    pa.add_argument("--seed", type=int, default=7)
    pa.add_argument("--preview", action="store_true")
    pa.add_argument("--no-video", action="store_true")
    pa.add_argument("--t-replan", type=float, default=T_REPLAN_DEFAULT,
                    help="Base TTC threshold [s] at full AMR speed. The actual "
                         "threshold scales with alpha_amr(v_actual), so a stationary "
                         "AMR has an effective window of 0 and never REPLANs.")
    pa.add_argument("--v-amr-typical", type=float, default=V_AMR_TYPICAL_DEFAULT,
                    help="Reference AMR speed at which alpha_amr saturates to 1.")
    pa.add_argument("--inject-stray", action="store_true",
                    help="Enable the engineered 'Loader' collision demo "
                         "(off by default -- the natural worker/AMR-B "
                         "encounter already exercises the collision path).")
    args = pa.parse_args(argv)

    preview = args.preview or args.no_video
    output_path = None if preview else Path(args.output)

    if not preview:
        print(f"Rendering {args.frames} frames with {args.workers} worker(s) "
              f"+ {args.amrs} AMR(s) -> {output_path}")
    saved = build_step_c_animation(
        output_path=output_path,
        num_frames=args.frames,
        num_workers=args.workers,
        num_amrs=args.amrs,
        preview=preview,
        seed=args.seed,
        t_replan=args.t_replan,
        v_amr_typical=args.v_amr_typical,
        inject_stray=args.inject_stray,
    )
    if saved is not None:
        print(f"Saved demo to {saved}")


if __name__ == "__main__":
    main()
