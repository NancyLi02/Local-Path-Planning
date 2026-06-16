"""
Advanced map-aware multi-worker trajectory prediction demo.

What this demo shows
--------------------
Per worker, every frame:
  1. Recursive Bayesian intent filter over a discrete set of factory goals
     (workstations). The belief is updated from direction-alignment and
     distance-to-goal evidence and smoothed in time.
  2. Particle rollout of N futures over a 5 s horizon under social-force
     dynamics (goal attraction, static-obstacle repulsion, damping, process
     noise, speed/acceleration limits).
  3. Kernel-density occupancy heatmap of where the worker is likely to be
     at the prediction horizon.
  4. Top-K mode trajectories: the mean rollout conditioned on each goal,
     colored by intent probability so the dominant intent stands out.

Run
---
    python human_predict.py                      # save mp4 to outputs/
    python human_predict.py --preview            # open a live window
    python human_predict.py --workers 3 --frames 90
    python human_predict.py --output my_demo.mp4
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, Rectangle


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAP_BOUNDS = (0.0, 20.0, 0.0, 12.0)  # xmin, xmax, ymin, ymax

# Each workstation is a static obstacle that also defines one candidate goal.
# The goal point sits INSIDE the workstation rectangle, close to the side that
# faces the central aisle (so workers naturally enter through that side).
# The workstation name is rendered on the opposite side of the rectangle so it
# does not collide with the goal star.
#   rect       : (x, y, w, h) of the obstacle in metres
#   goal       : (gx, gy) goal point inside the rectangle
#   name_pos   : (nx, ny) where to anchor the workstation name text
#   gid_offset : (dx, dy) where to anchor the "Gi" label relative to the goal
WORKSTATIONS: list[dict] = [
    dict(  # G1 — top-left, aisle is below: goal near bottom edge, name near top
        name="Assembly Cell",
        rect=(1.0, 8.2, 5.2, 2.7),
        goal=(3.6, 8.7),
        name_pos=(3.6, 10.4),
        gid_offset=(0.25, 0.35),
    ),
    dict(  # G2 — bottom-center, aisle is above: goal near top edge, name near bottom
        name="Battery Station",
        rect=(7.3, 0.7, 4.2, 2.3),
        goal=(9.4, 2.6),
        name_pos=(9.4, 1.1),
        gid_offset=(0.25, -0.45),
    ),
    dict(  # G3 — bottom-right, aisle is above
        name="Paint Buffer",
        rect=(13.3, 1.0, 4.7, 2.5),
        goal=(15.65, 3.0),
        name_pos=(15.65, 1.5),
        gid_offset=(0.25, -0.45),
    ),
    dict(  # G4 — top-right, aisle is below
        name="Body Shop",
        rect=(14.2, 8.6, 4.0, 2.0),
        goal=(16.2, 9.0),
        name_pos=(16.2, 10.25),
        gid_offset=(0.25, 0.32),
    ),
]

# Derived views for the predictor.
OBSTACLES: list[tuple[float, float, float, float, str]] = [
    (*ws["rect"], ws["name"]) for ws in WORKSTATIONS
]
GOALS = np.array([ws["goal"] for ws in WORKSTATIONS], dtype=float)


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


# ---------------------------------------------------------------------------
# Worker ground-truth generators
# ---------------------------------------------------------------------------


def _curved_path(
    start: np.ndarray,
    waypoints: list[np.ndarray],
    num_frames: int,
    dt: float,
    speed: float = 0.50,
    wobble: float = 0.08,
    seed: int = 0,
    stop_radius: float = 0.35,
) -> np.ndarray:
    """Walk through `waypoints` in order, then stop within `stop_radius` of the goal."""
    rng = np.random.default_rng(seed)
    pts = []
    p = start.astype(float).copy()
    v = np.zeros(2)
    wp_idx = 0
    arrived = False
    for k in range(num_frames):
        target = waypoints[min(wp_idx, len(waypoints) - 1)]
        to_t = target - p
        dist = np.linalg.norm(to_t) + 1e-6
        if dist < 0.8 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            target = waypoints[wp_idx]
            to_t = target - p
            dist = np.linalg.norm(to_t) + 1e-6
        # Stop once the final goal is reached.
        if wp_idx == len(waypoints) - 1 and dist < stop_radius:
            arrived = True
        if arrived:
            v *= 0.65                             # smooth stop, no sway
            p = p + v * dt
            pts.append(p.copy())
            continue
        desired = to_t / dist * speed
        sway = wobble * np.array([math.sin(k / 8.0 + seed), math.cos(k / 11.0 + seed)])
        v += 0.35 * (desired - v) * dt + sway * dt + rng.normal(scale=0.02, size=2) * dt
        sp = np.linalg.norm(v)
        if sp > speed * 1.25:
            v = v / sp * speed * 1.25
        p = p + v * dt
        pts.append(p.copy())
    return np.asarray(pts)


def make_workers(num_frames: int, dt: float, num_workers: int) -> list[dict]:
    """Build up to four scripted workers, each heading to a different workstation.

    Routes are designed so that mid-trajectory each worker passes through an
    ambiguous area where two goals are plausible, which makes the recursive
    Bayes intent filter visibly interesting.
    """
    g1, g2, g3, g4 = GOALS  # G1: Assembly, G2: Battery, G3: Paint, G4: Body
    scripts = [
        dict(  # bottom-left -> aisle -> Body Shop (G4): ambiguous G3 vs G4 mid-route
            name="Tech-1",
            color="#111111",
            start=np.array([2.0, 2.0]),
            waypoints=[np.array([11.0, 5.0]), g4],
            speed=0.50,
            seed=1,
        ),
        dict(  # bottom-right -> aisle -> Assembly Cell (G1): ambiguous G1 vs G2
            name="Tech-2",
            color="#7b3f00",
            start=np.array([18.5, 3.0]),
            waypoints=[np.array([12.0, 5.5]), g1],
            speed=0.50,
            seed=2,
        ),
        dict(  # top-left -> aisle -> Battery Station (G2): ambiguous G2 vs G3
            name="QC-1",
            color="#005f73",
            start=np.array([3.0, 11.0]),
            waypoints=[np.array([8.0, 7.0]), g2],
            speed=0.50,
            seed=3,
        ),
        dict(  # top-right -> aisle -> Paint Buffer (G3): ambiguous G2 vs G3
            name="Logistics",
            color="#3a0ca3",
            start=np.array([19.0, 11.0]),
            waypoints=[np.array([14.5, 7.0]), g3],
            speed=0.50,
            seed=4,
        ),
    ]
    workers = []
    for sc in scripts[:num_workers]:
        truth = _curved_path(
            sc["start"], sc["waypoints"], num_frames, dt,
            speed=sc["speed"], seed=sc["seed"],
        )
        workers.append(dict(name=sc["name"], color=sc["color"], truth=truth))
    return workers


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _draw_factory(ax) -> None:
    ax.set_xlim(MAP_BOUNDS[0], MAP_BOUNDS[1])
    ax.set_ylim(MAP_BOUNDS[2], MAP_BOUNDS[3])
    ax.set_aspect("equal")
    ax.set_facecolor("#f7fbff")
    ax.set_xlabel("Factory x [m]")
    ax.set_ylabel("Factory y [m]")
    ax.set_title(
        "Multi-worker trajectory prediction",
        fontsize=13, pad=10,
    )

    for ws in WORKSTATIONS:
        ox, oy, w, h = ws["rect"]
        ax.add_patch(Rectangle((ox, oy), w, h,
                               facecolor="#e8e8e8", edgecolor="#737373",
                               linewidth=1.2, zorder=0))
        nx, ny = ws["name_pos"]
        ax.text(nx, ny, ws["name"],
                ha="center", va="center", fontsize=9, color="#333333")

    ax.scatter(GOALS[:, 0], GOALS[:, 1], marker="*", s=220,
               c="#6a3d9a", edgecolors="white", linewidths=1.1, zorder=4,
               label="Candidate worker goals")
    for i, ws in enumerate(WORKSTATIONS):
        gx, gy = ws["goal"]
        dx, dy = ws["gid_offset"]
        ax.text(gx + dx, gy + dy, f"G{i+1}",
                fontsize=10, color="#4b1f6f", fontweight="bold")


def _kde_heatmap(particles: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                 bandwidth: float = 0.55) -> np.ndarray:
    """Cheap KDE over the final-step particles, evaluated on a coarse grid."""
    samples = particles[-1]                          # (N, 2)
    xs = grid_x[None, :, None] - samples[:, 0][:, None, None]
    ys = grid_y[None, None, :] - samples[:, 1][:, None, None]
    sq = xs * xs + ys * ys                           # (N, GX, GY)
    inv = 1.0 / (2.0 * bandwidth * bandwidth)
    vals = np.exp(-sq * inv).mean(axis=0)            # (GX, GY)
    return vals.T                                    # (GY, GX) for imshow


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def build_animation(
    output_path: Path | None,
    num_frames: int = 80,
    num_workers: int = 2,
    preview: bool = False,
    seed: int = 7,
):
    cfg = PredictorConfig(seed=seed)
    workers = make_workers(num_frames, cfg.dt, num_workers)
    predictors = [
        IntentParticlePredictor(
            GOALS, OBSTACLES, cfg,
            rng=np.random.default_rng(seed + 17 * i),
        )
        for i in range(len(workers))
    ]

    # ----- figure layout: map on left, intent bars on right -----
    fig = plt.figure(figsize=(15, 8.2), dpi=120)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3.4, 1.0],
        height_ratios=[14.0, 1.0],
        wspace=0.22, hspace=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    # Bottom legend only spans the map column so it cannot collide with the
    # x-axis label of the intent panel on the right.
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")

    _draw_factory(ax)

    # KDE grid (coarse for speed).
    grid_x = np.linspace(MAP_BOUNDS[0], MAP_BOUNDS[1], 80)
    grid_y = np.linspace(MAP_BOUNDS[2], MAP_BOUNDS[3], 48)
    heat_im = ax.imshow(
        np.zeros((len(grid_y), len(grid_x))),
        extent=(MAP_BOUNDS[0], MAP_BOUNDS[1], MAP_BOUNDS[2], MAP_BOUNDS[3]),
        origin="lower", cmap="Oranges", alpha=0.45, vmin=0.0, vmax=1.0,
        zorder=1,
    )

    # Per-worker visual handles.
    w_artists = []
    for w in workers:
        hist_line, = ax.plot([], [], "-", lw=2.0, color=w["color"],
                             label=f"{w['name']} observed", zorder=5)
        dot = Circle((0, 0), 0.22, facecolor=w["color"],
                     edgecolor="white", lw=1.0, zorder=6)
        ax.add_patch(dot)
        mean_line, = ax.plot([], [], "-", lw=2.2, color=w["color"],
                             alpha=0.75, zorder=4,
                             label=f"{w['name']} predicted mean")
        # One faint line per candidate goal for top-K modes.
        mode_lines = [
            ax.plot([], [], "--", lw=1.3, color=w["color"], alpha=0.0, zorder=3)[0]
            for _ in range(len(GOALS))
        ]
        # 95 % uncertainty ellipses along the horizon.
        ellipses_h = []
        for _ in range(cfg.horizon_steps):
            e = Ellipse((0, 0), 0.1, 0.1, angle=0, fill=False,
                        edgecolor=w["color"], alpha=0.0, lw=1.0, zorder=3)
            ax.add_patch(e)
            ellipses_h.append(e)
        w_artists.append(dict(
            hist=hist_line, dot=dot, mean=mean_line,
            modes=mode_lines, ellipses=ellipses_h,
        ))

    # Anchor the info box at the top-left of the map where there is empty
    # space above the obstacles, so it does not occlude any workstation label.
    info_text = ax.text(
        MAP_BOUNDS[0] + 0.4, MAP_BOUNDS[3] - 0.3, "",
        fontsize=9, color="#222",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        zorder=8,
        va="top", ha="left",
    )

    # Build a single horizontal legend at the bottom, outside the map area.
    legend_handles = []
    legend_labels = []
    # Goal marker proxy (matches the actual scatter)
    legend_handles.append(plt.Line2D(
        [0], [0], marker="*", color="w", markerfacecolor="#6a3d9a",
        markeredgecolor="white", markersize=12, linestyle="None",
    ))
    legend_labels.append("Candidate worker goals")
    # Heatmap proxy
    legend_handles.append(Rectangle((0, 0), 1, 1, facecolor="#fdae6b", alpha=0.7))
    legend_labels.append("5 s occupancy heatmap (KDE)")
    # Mode-trajectory proxy
    legend_handles.append(plt.Line2D([0], [0], linestyle="--", color="#555555", lw=1.5))
    legend_labels.append("Per-goal predicted mode (alpha = P(goal))")
    # Ellipse proxy
    legend_handles.append(plt.Line2D([0], [0], marker="o", color="#555555",
                                     markerfacecolor="none", markersize=10,
                                     linestyle="None"))
    legend_labels.append("95% uncertainty ellipses along 5 s horizon")
    # One entry per worker (observed + predicted-mean color)
    for w in workers:
        legend_handles.append(plt.Line2D([0], [0], color=w["color"], lw=2.4))
        legend_labels.append(f"{w['name']}: observed (solid) + mean prediction")

    ax_legend.legend(
        legend_handles, legend_labels,
        loc="center", ncol=3, frameon=True,
        framealpha=0.95, edgecolor="#cccccc", fontsize=9,
    )

    # ----- intent probability side panel (uses data coordinates per worker) -----
    ax_bar.set_title("Intent probabilities", fontsize=11, pad=8)
    ax_bar.set_xlim(0.0, 1.0)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_xticks([0.0, 0.5, 1.0])
    ax_bar.set_xlabel("P(goal | history)")
    ax_bar.set_yticks([])
    ax_bar.set_facecolor("#fafafa")
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    G = len(GOALS)
    W = len(workers)
    margin_top = 0.04
    # Leave more room at the bottom so the xlabel never collides with the bars.
    margin_bottom = 0.12
    gap_between_workers = 0.04
    usable = 1.0 - margin_top - margin_bottom - gap_between_workers * max(W - 1, 0)
    block_height = usable / W                          # height per worker block
    bar_height = block_height / (G + 1) * 0.85         # leave room for header
    header_height = block_height - G * bar_height
    label_left = 0.05
    bar_left = 0.18
    bar_full_width = 0.72
    bar_text_x = bar_left + bar_full_width + 0.02

    bar_groups = []
    for wi, w in enumerate(workers):
        block_top = 1.0 - margin_top - wi * (block_height + gap_between_workers)
        # Worker header label
        ax_bar.text(
            label_left, block_top - header_height * 0.55, w["name"],
            fontsize=10.5, color=w["color"], fontweight="bold",
            transform=ax_bar.transAxes, va="center",
        )
        bars = []
        for gi in range(G):
            row_top = block_top - header_height - gi * bar_height
            y = row_top - bar_height * 0.9            # bottom of bar
            rect = Rectangle(
                (bar_left, y), 0.0, bar_height * 0.78,
                facecolor=w["color"], alpha=0.7,
                transform=ax_bar.transAxes,
            )
            ax_bar.add_patch(rect)
            ax_bar.text(
                label_left, y + bar_height * 0.39, f"G{gi+1}",
                fontsize=8.5, color="#444",
                transform=ax_bar.transAxes, va="center",
            )
            txt = ax_bar.text(
                bar_text_x, y + bar_height * 0.39, "0.00",
                fontsize=8.5, color="#222",
                transform=ax_bar.transAxes, va="center", ha="left",
            )
            bars.append((rect, txt))
        bar_groups.append(bars)

    # ----- per-frame update -----
    def update(frame: int):
        heat = np.zeros((len(grid_y), len(grid_x)))

        for wi, (w, predictor, art) in enumerate(zip(workers, predictors, w_artists)):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]

            out = predictor.rollout(obs)
            belief = out["belief"]
            particles = out["particles"]
            mean_traj = out["mean"]
            mode_traj = out["mode_traj"]
            ellipses = out["ellipses"]

            art["hist"].set_data(w["truth"][: frame + 1, 0],
                                 w["truth"][: frame + 1, 1])
            art["dot"].center = tuple(w["truth"][frame])
            art["mean"].set_data(mean_traj[:, 0], mean_traj[:, 1])

            for gi, line in enumerate(art["modes"]):
                line.set_data(mode_traj[gi, :, 0], mode_traj[gi, :, 1])
                line.set_alpha(float(np.clip(belief[gi] * 1.3, 0.05, 0.95)))
                line.set_linewidth(1.2 + 2.4 * float(belief[gi]))

            for t, e in enumerate(art["ellipses"]):
                x, y, ww, hh, ang = ellipses[t]
                e.center = (x, y)
                e.width = ww
                e.height = hh
                e.angle = ang
                e.set_alpha(0.04 + 0.18 * t / cfg.horizon_steps)

            heat += _kde_heatmap(particles, grid_x, grid_y)

            bars = bar_groups[wi]
            for gi, (rect, txt) in enumerate(bars):
                rect.set_width(bar_full_width * float(belief[gi]))
                txt.set_text(f"{belief[gi]:.2f}")

        if heat.max() > 1e-6:
            heat = heat / heat.max()
        heat_im.set_data(heat)

        info_text.set_text(
            f"frame {frame+1}/{num_frames}    "
            f"horizon {cfg.horizon_steps*cfg.dt:.1f} s    "
            f"particles/worker {cfg.num_particles}\n"
            "method: recursive Bayes intent  +  social-force particle rollout  "
            "+  KDE occupancy"
        )

        if not preview and (frame % 5 == 0 or frame == num_frames - 1):
            print(f"  rendering frame {frame+1}/{num_frames}")

        artists = [heat_im, info_text]
        for art in w_artists:
            artists += [art["hist"], art["dot"], art["mean"]]
            artists += art["modes"] + art["ellipses"]
        for bars in bar_groups:
            for rect, txt in bars:
                artists += [rect, txt]
        return artists

    anim = FuncAnimation(fig, update, frames=num_frames, interval=90, blit=False)

    if preview:
        print("Preview mode: showing live window. Close it to exit.")
        plt.show()
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(fps=12, bitrate=2400)
        anim.save(output_path, writer=writer)
        saved = output_path
    except Exception as exc:                                # pragma: no cover
        print(f"  ffmpeg unavailable ({exc}); falling back to GIF")
        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=10))
        saved = gif_path
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_output() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "worker_prediction_demo.mp4"


def main(argv: list[str] | None = None) -> None:
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--output", type=str, default=str(_default_output()),
                    help="output path for the .mp4 (or .gif fallback)")
    pa.add_argument("--frames", type=int, default=80, help="number of simulation frames")
    pa.add_argument("--workers", type=int, default=2, choices=[1, 2, 3, 4],
                    help="number of workers in the scene")
    pa.add_argument("--seed", type=int, default=7)
    pa.add_argument("--preview", action="store_true",
                    help="open a live matplotlib window instead of rendering a file")
    pa.add_argument("--no-video", action="store_true",
                    help="alias for --preview")
    args = pa.parse_args(argv)

    preview = args.preview or args.no_video
    output_path = None if preview else Path(args.output)

    if not preview:
        print(f"Rendering {args.frames} frames with {args.workers} worker(s) "
              f"-> {output_path}")
    saved = build_animation(
        output_path=output_path,
        num_frames=args.frames,
        num_workers=args.workers,
        preview=preview,
        seed=args.seed,
    )
    if saved is not None:
        print(f"Saved demo to {saved}")


if __name__ == "__main__":
    main()
