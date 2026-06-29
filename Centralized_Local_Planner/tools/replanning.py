"""Step E (V0) -- TTC-priority sequential replanning with a space-time shield.

This is the deterministic (non-learned) safety baseline for the local
replanning layer. It sits on top of Steps A-D:

    A prediction -> B safety_inflation -> C affected_amr -> D conflict_cluster
                                                              |
                                                              v
                                              E (V0): this module

Per frame, for every active AMR (processed in ascending time-to-collision
order, i.e. the most urgent AMR first):

1.  **Candidate trajectory set.**  Because each AMR is rail-constrained (it can
    only move along its QR reference path), a local "trajectory" is a choice of
    forward speed.  We enumerate a small discrete set of speed factors applied
    to the planner-allowed speed: full / two-thirds / one-third / stop.

2.  **Space-time reservation shield.**  A candidate is admissible iff, over the
    shield horizon, its swept rail positions stay OUT of (a) every worker's
    Step-B hard safety lobe at the matching look-ahead step, and (b) the
    space-time footprints already reserved by higher-priority AMRs.

3.  The AMR commits to the FASTEST admissible candidate (maximise throughput),
    falls back to STOP if none is admissible, and reserves its chosen footprint
    so lower-priority AMRs yield to it.

The result is collision-free-by-construction motion (subject to the model):
the shield never lets an AMR drive into a worker no-go zone or into a
higher-priority peer's reserved space-time cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import point_in_polygon
from .affected_amr import AMR, ConflictResult


# Action labels (for rendering / metrics).
ACTION_GO = "GO"
ACTION_SLOW = "SLOW"
ACTION_STOP = "STOP"


@dataclass
class ReplanConfig:
    # Candidate speed factors (fraction of the planner-allowed speed), tried
    # fastest-first so the AMR keeps as much throughput as the shield allows.
    speed_factors: tuple[float, ...] = (1.0, 0.66, 0.33, 0.0)
    # How many look-ahead steps the shield must keep clear (<= predictor T).
    # Shorter = less conservative (fewer needless stops), longer = safer.
    shield_steps: int = 8
    # Min centre distance [m] between an AMR and a higher-priority AMR's
    # reserved space-time footprint.
    reserve_clearance: float = 1.0
    # Speed factor below which an action is labelled STOP / SLOW.
    slow_threshold: float = 0.95
    stop_threshold: float = 0.05


@dataclass
class SpaceTimeReservation:
    """Reservation table: per look-ahead step, a list of reserved (x, y)."""
    horizon: int
    clearance: float = 1.0
    _cells: list[list[np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._cells = [[] for _ in range(self.horizon)]

    def reserve(self, positions: np.ndarray) -> None:
        for t in range(min(self.horizon, len(positions))):
            self._cells[t].append(np.asarray(positions[t], dtype=float))

    def conflict(self, positions: np.ndarray) -> bool:
        for t in range(min(self.horizon, len(positions))):
            p = positions[t]
            for q in self._cells[t]:
                if float(np.linalg.norm(p - q)) < self.clearance:
                    return True
        return False


class V0Replanner:
    """TTC-priority sequential replanner + space-time reservation shield."""

    def __init__(self, cfg: ReplanConfig | None = None):
        self.cfg = cfg or ReplanConfig()
        # Diagnostics from the most recent plan() call.
        self.last_actions: dict[str, str] = {}
        self.last_speeds: dict[str, float] = {}

    # -- candidate rail trajectory at a given speed ------------------------
    def _rail_positions(self, amr: AMR, speed: float, dt: float, steps: int) -> np.ndarray:
        return np.array([
            amr.position_at(amr.progress + speed * (t + 1) * dt)
            for t in range(steps)
        ])

    def _candidate_safe(
        self,
        positions: np.ndarray,
        worker_data: list[dict],
        reservation: SpaceTimeReservation,
        steps: int,
    ) -> bool:
        # (a) worker hard no-go lobes, time-aligned
        for w in worker_data:
            hard = w["inflated"]
            for t in range(min(steps, len(hard))):
                if point_in_polygon(positions[t], hard[t]):
                    return False
        # (b) higher-priority AMR space-time reservations
        if reservation.conflict(positions):
            return False
        return True

    def _label(self, factor: float) -> str:
        if factor <= self.cfg.stop_threshold:
            return ACTION_STOP
        if factor < self.cfg.slow_threshold:
            return ACTION_SLOW
        return ACTION_GO

    def plan(
        self,
        amrs: list[AMR],
        results: dict[str, ConflictResult],
        worker_data: list[dict],
        frame: int,
        dt: float,
        horizon_T: int,
    ) -> dict[str, float]:
        """Choose + commit a speed for each active AMR. Mutates amr.actual_speed."""
        cfg = self.cfg
        steps = min(cfg.shield_steps, horizon_T)
        reservation = SpaceTimeReservation(horizon=steps, clearance=cfg.reserve_clearance)

        active = [a for a in amrs if a.is_active(frame)]

        # TTC-priority: most urgent (smallest ttc) first. AMRs without a
        # conflict result (or CLEAR) get ttc = +inf and are planned last.
        def ttc_of(a: AMR) -> float:
            r = results.get(a.name)
            return r.ttc if (r is not None and np.isfinite(r.ttc)) else float("inf")

        active.sort(key=ttc_of)

        self.last_actions = {}
        self.last_speeds = {}

        for amr in active:
            base_v = float(amr.actual_speed)   # planner-allowed (QR) speed cap
            chosen_v = 0.0
            chosen_factor = 0.0
            for factor in cfg.speed_factors:
                v = base_v * factor
                pos = self._rail_positions(amr, v, dt, steps)
                if self._candidate_safe(pos, worker_data, reservation, steps):
                    chosen_v = v
                    chosen_factor = factor
                    break
            # Commit: never exceed the planner-allowed speed.
            amr.actual_speed = min(base_v, chosen_v)
            # Reserve the committed footprint so lower-priority AMRs yield.
            reservation.reserve(self._rail_positions(amr, amr.actual_speed, dt, steps))

            self.last_speeds[amr.name] = amr.actual_speed
            self.last_actions[amr.name] = self._label(chosen_factor if base_v > 1e-9 else 0.0)

        return dict(self.last_speeds)
