"""Pre-computed Step-A/B worker data.

Workers are scripted and independent of the AMRs, so their prediction +
inflation can be computed once per seed and reused by every planner, config
and RL rollout. The particle predictor is the slowest part of the pipeline,
so this is what makes sweeps and training tractable.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from Centralized_Local_Planner.tools.factory_map import GOALS, OBSTACLES
from Centralized_Local_Planner.tools.geometry import safety_tube_polygon
from Centralized_Local_Planner.tools.prediction import IntentParticlePredictor, PredictorConfig
from Centralized_Local_Planner.tools.safety_inflation import (
    SafetyInflationConfig, SafetyInflationModel,
)
from Centralized_Local_Planner.tools.scenario import make_workers

_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "step_e_v1"


def worker_frames(num_frames: int = 420, num_workers: int = 2, seed: int = 0,
                  use_disk: bool = True) -> list[list[dict]]:
    """``frames[f]`` = the Step-A/B ``worker_data`` list for frame ``f``."""
    tag = hashlib.md5(f"{num_frames}_{num_workers}_{seed}".encode()).hexdigest()[:12]
    path = _CACHE / f"workers_{tag}.pkl"
    if use_disk and path.exists():
        return pickle.loads(path.read_bytes())

    cfg = PredictorConfig(seed=seed)
    safety = SafetyInflationModel(SafetyInflationConfig())
    workers = make_workers(num_frames, cfg.dt, num_workers)
    predictors = [IntentParticlePredictor(GOALS, OBSTACLES, cfg,
                                          rng=np.random.default_rng(seed + 17 * i))
                  for i in range(len(workers))]
    frames = []
    for f in range(num_frames):
        out = []
        for w, predictor in zip(workers, predictors):
            obs = w["truth"][max(0, f - 8): f + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            pred = predictor.rollout(obs)
            hard, soft, _ = safety.inflate_all(pred["ellipses"], cfg.dt,
                                               pred["belief"], mean_traj=pred["mean"])
            out.append(dict(name=w["name"], color=w["color"], inflated=hard, soft=soft,
                            centers=pred["ellipses"][:, :2], ellipses=pred["ellipses"],
                            tube=safety_tube_polygon(hard)))
        frames.append(out)
    if use_disk:
        _CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(frames))
    return frames
