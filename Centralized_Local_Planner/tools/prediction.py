"""Step A -- worker trajectory prediction.

Recursive-Bayes intent filter over factory goals + social-force particle
rollout over a 5 s horizon (IntentParticlePredictor)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .factory_map import MAP_BOUNDS


@dataclass
class PredictorConfig:
    dt: float = 0.2
    horizon_steps: int = 25            # 5 s lookahead
    num_particles: int = 600
    max_speed: float = 0.85            # m/s cap for particle rollout
    max_acc: float = 2.4
    goal_gain: float = 1.15
    damping: float = 0.42
    static_repulsion_gain: float = 1.8
    static_repulsion_sigma: float = 0.85
    process_noise: float = 0.10
    intent_smoothing: float = 0.55     # recursive belief smoothing in [0,1]
    seed: int = 7


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class IntentParticlePredictor:
    """Recursive Bayesian intent + social-force particle rollout."""

    def __init__(
        self,
        goals: np.ndarray,
        obstacles: Sequence[tuple[float, float, float, float]],
        cfg: PredictorConfig,
        rng: np.random.Generator | None = None,
    ):
        self.goals = np.asarray(goals, dtype=float)
        self.obstacles = [(ox, oy, w, h) for (ox, oy, w, h, *_) in obstacles]
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        self.belief = np.full(len(self.goals), 1.0 / len(self.goals))

    # ---- intent inference -------------------------------------------------

    def _instant_likelihood(self, obs: np.ndarray) -> np.ndarray:
        """P(observation | goal) from motion alignment + distance-to-goal."""
        p = obs[-1]
        look_back = min(4, len(obs) - 1)
        v = (obs[-1] - obs[-1 - look_back]) / max(self.cfg.dt * look_back, 1e-6)
        speed = np.linalg.norm(v) + 1e-6
        v_dir = v / speed

        scores = []
        for g in self.goals:
            to_g = g - p
            dist = np.linalg.norm(to_g) + 1e-6
            dir_g = to_g / dist
            alignment = float(np.dot(v_dir, dir_g))
            # Confidence scales with current speed (stationary -> uninformative).
            scores.append(2.6 * alignment * min(1.0, speed / 0.4) - 0.11 * dist)
        scores = np.asarray(scores)
        scores -= scores.max()
        lik = np.exp(scores)
        return lik / lik.sum()

    def update_belief(self, obs: np.ndarray) -> np.ndarray:
        """Recursive Bayes with exponential smoothing for stability."""
        lik = self._instant_likelihood(obs)
        posterior = self.belief * lik
        posterior = posterior / posterior.sum()
        alpha = self.cfg.intent_smoothing
        self.belief = alpha * posterior + (1.0 - alpha) * self.belief
        self.belief = self.belief / self.belief.sum()
        return self.belief

    # ---- force fields -----------------------------------------------------

    def _static_repulsion(
        self,
        pos: np.ndarray,
        skip_idx: np.ndarray | None = None,
    ) -> np.ndarray:
        """Soft repulsion from every workstation rectangle.

        ``skip_idx[i]`` is the workstation index that particle ``i`` is heading
        into; that workstation is excluded from its own repulsion so the
        particle can actually reach a goal placed inside the rectangle.
        """
        force = np.zeros_like(pos)
        for oi, (ox, oy, w, h) in enumerate(self.obstacles):
            closest_x = np.clip(pos[:, 0], ox, ox + w)
            closest_y = np.clip(pos[:, 1], oy, oy + h)
            dxy = pos - np.column_stack([closest_x, closest_y])
            dist = np.linalg.norm(dxy, axis=1) + 1e-6
            direction = dxy / dist[:, None]
            mag = self.cfg.static_repulsion_gain * np.exp(
                -dist / self.cfg.static_repulsion_sigma
            )
            if skip_idx is not None:
                mag = np.where(skip_idx == oi, 0.0, mag)
            force += direction * mag[:, None]
        return force

    # ---- rollout ----------------------------------------------------------

    def rollout(self, obs: np.ndarray) -> dict:
        """One-shot multi-goal rollout.

        Returns dict with keys:
            belief        (G,)
            particles     (T, N, 2)
            mean          (T, 2)        weighted by belief
            mode_traj     (G, T, 2)     mean conditioned on each goal
            ellipses      (T, 5)        x, y, w, h, angle (overall)
            goal_idx      (N,)
        """
        cfg = self.cfg
        belief = self.update_belief(obs)
        n = cfg.num_particles

        current = obs[-1]
        prev = obs[-2] if len(obs) >= 2 else obs[-1]
        mean_v = (current - prev) / cfg.dt

        pos = current + self.rng.normal(scale=0.04, size=(n, 2))
        vel = mean_v + self.rng.normal(scale=0.16, size=(n, 2))

        goal_idx = self.rng.choice(len(self.goals), size=n, p=belief)
        particle_goals = self.goals[goal_idx]

        particles = np.empty((cfg.horizon_steps, n, 2))
        for t in range(cfg.horizon_steps):
            to_goal = particle_goals - pos
            dist = np.linalg.norm(to_goal, axis=1) + 1e-6
            desired_dir = to_goal / dist[:, None]
            desired_speed = np.minimum(cfg.max_speed, 0.48 * dist)
            desired_v = desired_dir * desired_speed[:, None]

            acc = cfg.goal_gain * (desired_v - vel) - cfg.damping * vel
            # Each particle ignores the workstation it is currently heading to
            # so it can actually reach a goal placed inside the rectangle.
            acc += self._static_repulsion(pos, skip_idx=goal_idx)
            acc += self.rng.normal(scale=cfg.process_noise, size=(n, 2))

            acc_norm = np.linalg.norm(acc, axis=1) + 1e-6
            acc *= np.minimum(1.0, cfg.max_acc / acc_norm)[:, None]
            vel = vel + acc * cfg.dt
            sp = np.linalg.norm(vel, axis=1) + 1e-6
            vel *= np.minimum(1.0, cfg.max_speed / sp)[:, None]
            pos = pos + vel * cfg.dt

            pos[:, 0] = np.clip(pos[:, 0], MAP_BOUNDS[0] + 0.4, MAP_BOUNDS[1] - 0.4)
            pos[:, 1] = np.clip(pos[:, 1], MAP_BOUNDS[2] + 0.4, MAP_BOUNDS[3] - 0.4)
            particles[t] = pos

        # Per-goal modes (mean trajectory of particles assigned to each goal).
        mode_traj = np.zeros((len(self.goals), cfg.horizon_steps, 2))
        for gi in range(len(self.goals)):
            mask = goal_idx == gi
            if mask.sum() >= 5:
                mode_traj[gi] = particles[:, mask, :].mean(axis=1)
            else:
                mode_traj[gi] = particles.mean(axis=1)

        mean_traj = particles.mean(axis=1)

        ellipses = np.array([self._ellipse_params(particles[t]) for t in range(cfg.horizon_steps)])
        return {
            "belief": belief,
            "particles": particles,
            "mean": mean_traj,
            "mode_traj": mode_traj,
            "ellipses": ellipses,
            "goal_idx": goal_idx,
        }

    @staticmethod
    def _ellipse_params(samples: np.ndarray) -> np.ndarray:
        mu = samples.mean(axis=0)
        cov = np.cov(samples.T) + 1e-4 * np.eye(2)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # 95 % confidence for a 2-D Gaussian
        scale = 2.448
        width, height = 2 * scale * np.sqrt(np.maximum(vals, 0.0))
        angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
        return np.array([mu[0], mu[1], width, height, angle])
