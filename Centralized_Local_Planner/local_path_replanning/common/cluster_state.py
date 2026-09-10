"""Data contract between the factory simulator (Steps A-D) and local path replanning.

The planner modules (`observation_builder`, `trajectory_rollout`,
`safety_shield`, `v0_planner`, `v1_planner`) only ever see the plain
structures defined here, so the simulator can be swapped for a real robot
stack by re-implementing this one file.

Frames
------
* world frame     : metres, the factory map frame.
* path frame      : (s, l) = arc-length along the AMR reference path and signed
                    lateral offset (left of the path tangent is positive).
* AMR local frame : x forward (along the AMR heading), y left.
"""
from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class PathRef(Protocol):
    """Reference-path provider (the simulator ``AMR`` satisfies this)."""
    total_length: float
    def position_at(self, s: float) -> np.ndarray: ...
    def heading_at(self, s: float) -> float: ...


# ---------------------------------------------------------------------------
# Cached reference path (vectorised path-frame lookups)
# ---------------------------------------------------------------------------

class PathTable:
    """Densely sampled reference path -> O(1) vectorised (s, lat) lookups.

    The planner evaluates thousands of path points per frame (six candidate
    trajectories x thirty samples x N AMRs); resampling the polyline in Python
    each time dominated the runtime, so every path is tabulated once.
    """

    _cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

    def __init__(self, ref: PathRef, ds: float = 0.02):
        self.total_length = float(ref.total_length)
        n = max(int(math.ceil(self.total_length / ds)) + 1, 2)
        self.s = np.linspace(0.0, self.total_length, n)
        self.pos = np.array([ref.position_at(float(s)) for s in self.s])
        d = np.gradient(self.pos, axis=0)
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        self.tan = d / np.maximum(nrm, 1e-9)
        self.nor = np.stack([-self.tan[:, 1], self.tan[:, 0]], axis=1)
        self.head = np.arctan2(self.tan[:, 1], self.tan[:, 0])

    @classmethod
    def get(cls, ref: PathRef) -> "PathTable":
        table = cls._cache.get(ref)
        if table is None:
            table = cls(ref)
            cls._cache[ref] = table
        return table

    def _idx(self, s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.total_length)
        return s / max(self.s[1] - self.s[0], 1e-12)

    def position(self, s, lat=0.0) -> np.ndarray:
        """Vectorised world position at arc-length ``s`` offset ``lat`` left."""
        f = self._idx(s)
        i0 = np.floor(f).astype(int)
        i1 = np.minimum(i0 + 1, len(self.s) - 1)
        w = (f - i0)[..., None]
        pos = self.pos[i0] * (1 - w) + self.pos[i1] * w
        nor = self.nor[i0] * (1 - w) + self.nor[i1] * w
        lat = np.asarray(lat, dtype=float)
        return pos + nor * (lat[..., None] if lat.ndim else lat)

    def heading(self, s) -> np.ndarray:
        f = self._idx(s)
        return self.head[np.clip(np.round(f).astype(int), 0, len(self.s) - 1)]

    def tangent(self, s) -> np.ndarray:
        f = self._idx(s)
        return self.tan[np.clip(np.round(f).astype(int), 0, len(self.s) - 1)]

    def normal(self, s) -> np.ndarray:
        f = self._idx(s)
        return self.nor[np.clip(np.round(f).astype(int), 0, len(self.s) - 1)]


# ---------------------------------------------------------------------------
# Path-frame helpers
# ---------------------------------------------------------------------------

def tangent_at(ref: PathRef, s: float) -> np.ndarray:
    h = ref.heading_at(float(np.clip(s, 0.0, ref.total_length)))
    return np.array([math.cos(h), math.sin(h)])


def normal_at(ref: PathRef, s: float) -> np.ndarray:
    """Left normal of the path tangent."""
    t = tangent_at(ref, s)
    return np.array([-t[1], t[0]])


def path_point(ref: PathRef, s: float, lat: float = 0.0) -> np.ndarray:
    """World point at arc-length ``s`` offset by ``lat`` metres to the left."""
    s = float(np.clip(s, 0.0, ref.total_length))
    return ref.position_at(s) + lat * normal_at(ref, s)


def project_onto_path(ref: PathRef, xy: np.ndarray, s_hint: float = 0.0,
                      window: float = 4.0, coarse: float = 0.20,
                      fine: float = 0.02) -> tuple[float, float]:
    """Project a world point onto the reference path -> (s, signed lateral).

    Searches a +/- ``window`` metre band around ``s_hint`` (coarse then fine),
    which is both fast and stable for self-intersecting factory routes.
    """
    xy = np.asarray(xy, dtype=float)
    lo = max(0.0, s_hint - window)
    hi = min(ref.total_length, s_hint + window)
    if hi <= lo:
        lo, hi = 0.0, ref.total_length

    def best_in(a: float, b: float, step: float) -> float:
        n = max(int(math.ceil((b - a) / max(step, 1e-6))), 1)
        cand = np.linspace(a, b, n + 1)
        d = [float(np.linalg.norm(ref.position_at(s) - xy)) for s in cand]
        return float(cand[int(np.argmin(d))])

    s = best_in(lo, hi, coarse)
    s = best_in(max(lo, s - coarse), min(hi, s + coarse), fine)
    lat = float(np.dot(xy - ref.position_at(s), normal_at(ref, s)))
    return s, lat


# ---------------------------------------------------------------------------
# Worker prediction (Step A + Step B output)
# ---------------------------------------------------------------------------

@dataclass
class WorkerPrediction:
    """One worker's predicted occupied tube over the prediction horizon."""
    name: str
    centers: np.ndarray            # (T, 2) predicted mean positions
    hard_lobes: np.ndarray         # (T, K, 2) NO-GO polygons
    soft_lobes: np.ndarray         # (T, K, 2) slowdown polygons
    ellipses: np.ndarray | None = None   # (T, 5) [x, y, width, height, angle_deg]
    dt: float = 0.2

    @property
    def horizon(self) -> int:
        return int(len(self.centers))

    def index_at(self, t_sec: float) -> int:
        """Prediction index whose look-ahead time is closest to ``t_sec``."""
        i = int(round(t_sec / max(self.dt, 1e-9))) - 1
        return int(np.clip(i, 0, self.horizon - 1))

    def sigma_at(self, idx: int) -> tuple[float, float]:
        if self.ellipses is None:
            return 0.0, 0.0
        e = self.ellipses[int(np.clip(idx, 0, len(self.ellipses) - 1))]
        return float(e[2]) * 0.5, float(e[3]) * 0.5


def worker_predictions_from_step_b(worker_data: Sequence[dict],
                                   dt: float = 0.2) -> list[WorkerPrediction]:
    """Adapt the Step-A/B per-frame ``worker_data`` dicts used by the demos."""
    out = []
    for w in worker_data:
        out.append(WorkerPrediction(
            name=w.get("name", "worker"),
            centers=np.asarray(w["centers"], dtype=float),
            hard_lobes=np.asarray(w["inflated"], dtype=float),
            soft_lobes=np.asarray(w["soft"], dtype=float),
            ellipses=np.asarray(w["ellipses"], dtype=float) if "ellipses" in w else None,
            dt=dt,
        ))
    return out


# ---------------------------------------------------------------------------
# Static map
# ---------------------------------------------------------------------------

@dataclass
class MapData:
    obstacles: list[tuple[float, float, float, float]]   # (x, y, w, h)
    bounds: tuple[float, float, float, float]            # xmin, xmax, ymin, ymax

    @staticmethod
    def from_factory(obstacles, bounds) -> "MapData":
        rects = [(float(o[0]), float(o[1]), float(o[2]), float(o[3])) for o in obstacles]
        return MapData(obstacles=rects, bounds=tuple(float(b) for b in bounds))


# ---------------------------------------------------------------------------
# Cluster agent
# ---------------------------------------------------------------------------

@dataclass
class ClusterAgent:
    """One AMR of a conflict cluster, as seen by the local replanner."""
    id: str
    ref: PathRef                    # global reference path
    s: float                        # arc-length progress
    lat: float                      # signed lateral offset from the path
    speed: float
    heading: float
    accel: float = 0.0
    omega: float = 0.0
    goal_s: float = 0.0             # local goal arc-length (downstream rejoin)
    ttc: float = float("inf")
    task_priority: float = 0.5      # 0..1, higher = more important task
    braking_risk: float = 0.0       # 0..1, how badly a hard stop would hurt
    affected: bool = False          # Step-C REPLAN / SLOWDOWN flag
    radius: float = 0.45
    v_cap: float = 0.35             # planner-allowed speed cap this frame
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.table = PathTable.get(self.ref)

    # -- geometry -----------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.path_position(self.s, self.lat)

    def path_position(self, s, lat=0.0) -> np.ndarray:
        return self.table.position(s, lat)

    def path_heading(self, s) -> np.ndarray:
        return self.table.heading(s)

    def tangent(self, s: float | None = None) -> np.ndarray:
        return self.table.tangent(self.s if s is None else s)

    def normal(self, s: float | None = None) -> np.ndarray:
        return self.table.normal(self.s if s is None else s)

    @property
    def total_length(self) -> float:
        return float(self.ref.total_length)

    @property
    def heading_rel(self) -> float:
        """Heading relative to the path tangent, wrapped to [-pi, pi]."""
        d = self.heading - float(self.path_heading(self.s))
        return float((d + math.pi) % (2 * math.pi) - math.pi)

    def to_local(self, world_xy: np.ndarray) -> np.ndarray:
        """World point -> AMR local frame (x forward, y left)."""
        d = np.asarray(world_xy, dtype=float) - self.position
        c, sn = math.cos(-self.heading), math.sin(-self.heading)
        return np.array([d[0] * c - d[1] * sn, d[0] * sn + d[1] * c])

    def rail_rollout(self, dt: float, steps: int, speed: float | None = None) -> np.ndarray:
        """Positions if the AMR simply kept following its path at ``speed``."""
        v = self.speed if speed is None else speed
        s = self.s + v * dt * np.arange(1, steps + 1)
        return self.path_position(s, np.full(steps, self.lat))
