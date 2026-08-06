"""Fast centralized fleet environment for V1 (attention RL behind the V0 shield).

Speed tricks that make RL training tractable on top of the slow particle
predictor:

1. Workers are scripted & independent of the AMRs -> their Step-A/B safety
   lobes are precomputed once per seed.
2. AMRs are rail-constrained -> a future position is a function of arc-length.
   We precompute a boolean **no-go table** `nogo[amr][frame][t][arclen_bucket]`
   (vectorised point-in-polygon, polygons downsampled), cached to disk. Both the
   shield projection AND the observation conflict features then collapse to
   O(steps) array lookups -- no per-step Python point-in-polygon at all.

Action = per-AMR desired speed factor in [0,1], projected through the SAME
space-time reservation shield as V0 (safety guaranteed by construction). The
policy only learns coordination/efficiency; V0 == "always propose factor 1".
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..tools.prediction import PredictorConfig, IntentParticlePredictor
from ..tools.safety_inflation import SafetyInflationConfig, SafetyInflationModel
from ..tools.factory_map import GOALS, OBSTACLES, MAP_BOUNDS
from ..tools.scenario import make_workers, make_amrs
from ..tools.affected_amr import CentralizedPlanner, amr_human_collision
from ..tools.replanning import SpaceTimeReservation

OBS_DIM = 12
_WORKER_COLLISION_DIST = 0.55
_ARCLEN_STEP = 0.08            # m, no-go table arc-length resolution
_POLY_STRIDE = 2              # downsample lobe polygons (64 -> 32 verts) for the table
_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


@dataclass
class FleetEnvConfig:
    num_frames: int = 360
    num_workers: int = 2
    num_amrs: int = 6
    shield_steps: int = 25
    reserve_clearance: float = 1.0
    speed_levels: int = 11
    w_progress: float = 1.0
    w_stop: float = 0.02
    w_time: float = 0.004
    w_done: float = 0.5
    w_clear: float = 0.05


# ---------------------------------------------------------------------------
# Vectorised geometry
# ---------------------------------------------------------------------------

def _points_in_poly(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray-cast point-in-polygon, many points vs one polygon. (M,2)->(M,)bool."""
    x = points[:, 0]; y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    K = len(poly); j = K - 1
    for i in range(K):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


# ---------------------------------------------------------------------------
# Per-seed caches
# ---------------------------------------------------------------------------

_WORKER_CACHE: dict[tuple, tuple] = {}
_NOGO_CACHE: dict[tuple, list] = {}
_RAIL_CACHE: dict[str, tuple] = {}


def precompute_worker_data(num_frames, num_workers, seed):
    key = (num_frames, num_workers, seed)
    if key in _WORKER_CACHE:
        return _WORKER_CACHE[key]
    cfg = PredictorConfig(seed=seed)
    safety = SafetyInflationModel(SafetyInflationConfig())
    workers = make_workers(num_frames, cfg.dt, num_workers)
    predictors = [IntentParticlePredictor(GOALS, OBSTACLES, cfg,
                  rng=np.random.default_rng(seed + 17 * i)) for i in range(len(workers))]
    frames = []
    for f in range(num_frames):
        wd = []
        for w, predictor in zip(workers, predictors):
            obs_start = max(0, f - 8)
            obs = w["truth"][obs_start: f + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            out = predictor.rollout(obs)
            hard, soft, _ = safety.inflate_all(
                out["ellipses"], cfg.dt, out["belief"], mean_traj=out["mean"])
            wd.append(dict(name=w["name"], inflated=hard, soft=soft,
                           centers=out["ellipses"][:, :2]))
        frames.append(wd)
    result = (frames, workers, cfg.dt, cfg.horizon_steps)
    _WORKER_CACHE[key] = result
    return result


def _rail_grid(amr):
    if amr.name not in _RAIL_CACHE:
        n = max(int(amr.total_length / _ARCLEN_STEP) + 1, 2)
        grid = np.linspace(0.0, amr.total_length, n)
        pts = np.array([amr.position_at(s) for s in grid])
        _RAIL_CACHE[amr.name] = (grid, pts)
    return _RAIL_CACHE[amr.name]


def precompute_nogo(worker_frames, amrs, shield_steps, num_frames, key):
    if key in _NOGO_CACHE:
        return _NOGO_CACHE[key]
    tag = hashlib.md5(str(key).encode()).hexdigest()[:12]
    cache_file = _CACHE_DIR / f"nogo_{tag}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        tables = [data[f"t{i}"] for i in range(len(amrs))]
        _NOGO_CACHE[key] = tables
        return tables

    tables = []
    for amr in amrs:
        _, rail_pts = _rail_grid(amr)
        tab = np.zeros((num_frames, shield_steps, len(rail_pts)), dtype=bool)
        for f in range(num_frames):
            wd = worker_frames[f]
            for t in range(shield_steps):
                hit = np.zeros(len(rail_pts), dtype=bool)
                for w in wd:
                    hit |= _points_in_poly(rail_pts, w["inflated"][t][::_POLY_STRIDE])
                tab[f, t] = hit
        tables.append(tab)
    _CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache_file, **{f"t{i}": tables[i] for i in range(len(tables))})
    _NOGO_CACHE[key] = tables
    return tables


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FleetEnv:
    """obs (num_amrs, OBS_DIM); act (num_amrs,) in [0,1] desired speed factor."""

    def __init__(self, cfg: FleetEnvConfig | None = None):
        self.cfg = cfg or FleetEnvConfig()
        self.N = self.cfg.num_amrs
        self._diag = float(np.hypot(MAP_BOUNDS[1] - MAP_BOUNDS[0],
                                    MAP_BOUNDS[3] - MAP_BOUNDS[2]))

    def reset(self, seed: int):
        c = self.cfg
        self.worker_frames, self.workers, self.dt, self.horizon_T = \
            precompute_worker_data(c.num_frames, c.num_workers, seed)
        self.steps = min(c.shield_steps, self.horizon_T)
        self.amrs = make_amrs(c.num_amrs)
        self.nogo = precompute_nogo(self.worker_frames, self.amrs, self.steps, c.num_frames,
                                    (c.num_frames, c.num_workers, seed, c.num_amrs, self.steps))
        self.rail = [_rail_grid(a) for a in self.amrs]
        self.planner = CentralizedPlanner(dt=self.dt)
        self.frame = 0
        self._prev_progress = np.array([a.progress for a in self.amrs])
        self._prev_done = np.array([a.is_done() for a in self.amrs])
        return self._observe()

    # -- table-based shield ----------------------------------------------
    def _hard_mask(self, i, speed):
        amr = self.amrs[i]; tab = self.nogo[i][self.frame]; nb = tab.shape[1]
        mask = np.zeros(self.steps, dtype=bool)
        for t in range(self.steps):
            k = int(round((amr.progress + speed * (t + 1) * self.dt) / _ARCLEN_STEP))
            if 0 <= k < nb:
                mask[t] = tab[t, k]
        return mask

    def _rail_positions(self, amr, speed):
        return np.array([amr.position_at(amr.progress + speed * (t + 1) * self.dt)
                         for t in range(self.steps)])

    def _max_safe_factor(self, i, base_v, reservation):
        amr = self.amrs[i]
        for factor in np.linspace(1.0, 0.0, self.cfg.speed_levels):
            if self._hard_mask(i, base_v * factor).any():
                continue
            if reservation is not None and reservation.conflict(
                    self._rail_positions(amr, base_v * factor)):
                continue
            return float(factor)
        return 0.0

    # -- observation ------------------------------------------------------
    def _observe(self):
        c = self.cfg
        f = min(self.frame, c.num_frames - 1)
        self.planner.resolve(self.amrs, self.frame)
        obs = np.zeros((self.N, OBS_DIM), dtype=np.float32)
        self._active_mask = np.zeros(self.N, dtype=bool)
        self._cur = []
        horizon_s = self.steps * self.dt
        for i, amr in enumerate(self.amrs):
            active = amr.is_active(self.frame)
            base_v = float(amr.actual_speed)
            ttc_n, hard_frac, headroom, ttc = 1.0, 0.0, 0.0, float("inf")
            d_worker, bx, by = 1.0, 0.0, 0.0
            if active:
                hmask = self._hard_mask(i, max(base_v, 1e-9))
                hard_frac = float(hmask.mean())
                if hmask.any():
                    ttc = (int(np.argmax(hmask)) + 1) * self.dt
                    ttc_n = ttc / horizon_s
                headroom = self._max_safe_factor(i, max(base_v, 1e-9), reservation=None)
                p = amr.position_at(amr.progress)
                best = None
                for w in self.workers:
                    wp = w["truth"][f] if f < len(w["truth"]) else w["truth"][-1]
                    dd = float(np.linalg.norm(p - wp))
                    if best is None or dd < best[0]:
                        best = (dd, wp)
                d_worker = min(best[0] / self._diag, 1.0)
                rel = best[1] - p
                nn = float(np.linalg.norm(rel)) + 1e-9
                bx, by = float(rel[0] / nn), float(rel[1] / nn)
            pos = amr.position_at(amr.progress)
            obs[i] = [
                1.0 if active else 0.0,
                amr.progress / max(amr.total_length, 1e-9),
                base_v / max(amr.commanded_speed, 1e-9),
                1.0 if amr.waiting_for else 0.0,
                ttc_n, hard_frac, headroom,
                d_worker, bx, by,
                (pos[0] - MAP_BOUNDS[0]) / (MAP_BOUNDS[1] - MAP_BOUNDS[0]),
                (pos[1] - MAP_BOUNDS[2]) / (MAP_BOUNDS[3] - MAP_BOUNDS[2]),
            ]
            self._active_mask[i] = active
            self._cur.append(dict(base_v=base_v, ttc=ttc))
        return obs

    @property
    def active_mask(self):
        return self._active_mask.copy()

    # -- step -------------------------------------------------------------
    def step(self, action: np.ndarray):
        c = self.cfg
        f = min(self.frame, c.num_frames - 1)
        reservation = SpaceTimeReservation(horizon=self.steps, clearance=c.reserve_clearance)
        action = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
        order = sorted([i for i in range(self.N) if self._active_mask[i]],
                       key=lambda i: self._cur[i]["ttc"])
        n_stop = 0
        for i in order:
            amr = self.amrs[i]
            base_v = self._cur[i]["base_v"]
            max_factor = self._max_safe_factor(i, max(base_v, 1e-9), reservation)
            # Action slack: action in (0,1) maps to a desired factor in (0,1.25)
            # so the policy CAN request the full max-safe speed (== V0) and is
            # not structurally capped below it by the open-interval Beta.
            desired = float(action[i]) * 1.25
            v = min(desired, max_factor) * base_v
            amr.actual_speed = v
            reservation.reserve(self._rail_positions(amr, v))
            if v < 1e-6:
                n_stop += 1

        for amr in self.amrs:
            if amr.is_active(self.frame):
                hit, hw = amr_human_collision(amr, self.frame, self.workers, _WORKER_COLLISION_DIST)
                if hit:
                    amr.mark_collision(self.frame, hw)
                elif amr.is_active(self.frame):
                    amr.step(self.dt)

        progress = np.array([a.progress for a in self.amrs])
        done_now = np.array([a.is_done() for a in self.amrs])
        advance = float(np.sum(progress - self._prev_progress))
        newly_done = int(np.sum(done_now & ~self._prev_done))
        n_active = int(np.sum([a.is_active(self.frame) for a in self.amrs]))
        clear_bonus = 0.0
        for a in self.amrs:
            if a.is_active(self.frame):
                p = a.position_at(a.progress)
                dmin = min(float(np.linalg.norm(p - (w["truth"][f] if f < len(w["truth"]) else w["truth"][-1])))
                           for w in self.workers)
                clear_bonus += min(dmin, 1.5) / 1.5
        reward = (c.w_progress * advance + c.w_done * newly_done
                  - c.w_stop * n_stop - c.w_time * n_active
                  + c.w_clear * clear_bonus / max(self.N, 1))

        self._prev_progress = progress
        self._prev_done = done_now
        self.frame += 1
        collided = int(np.sum([a.collided for a in self.amrs]))
        done = (self.frame >= c.num_frames) or all((a.is_done() or a.collided) for a in self.amrs)
        obs = self._observe() if not done else np.zeros((self.N, OBS_DIM), dtype=np.float32)
        info = dict(completion=float(np.mean(done_now)),
                    progress=float(np.mean(progress / np.array([a.total_length for a in self.amrs]))),
                    collided=collided, n_stop=n_stop)
        return obs, float(reward), done, info
