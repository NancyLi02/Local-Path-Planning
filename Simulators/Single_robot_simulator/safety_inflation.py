"""
Safety Inflation Model -- Step B of the centralized local planning core.

Inputs
------
Per-worker outputs from `human_predict.IntentParticlePredictor.rollout`
(Step A: belief, particle rollout, 95 % uncertainty ellipses, mean traj).

What it does
------------
For every lookahead step ``t`` along the 5 s horizon, builds a per-worker
*inflated* occupancy region that an AMR must avoid. Two design choices set
this version apart from a naive "fat circle":

1.  Velocity gating (worker speed).  Each dynamic margin is multiplied by
    a smoothstep ``alpha_h(v_h(t))`` so the entire tube collapses to the
    bare body radius when the worker is at rest:

        alpha_h(v_h) = smoothstep(v_h / v_full_walk)
        R_dyn(t)     = alpha_h * ( R_latency + R_braking
                                 + R_reaction * (1 + H(belief)/log G)
                                 + alpha_forecast * t )
        v_h(t) is read from the Step-A mean trajectory.

2.  Forward-bias (worker heading).  Walking humans pull most of their
    uncertainty IN FRONT of them; the area immediately behind them is
    relatively safe -- they will not back into it. The safety zone is
    therefore an asymmetric teardrop, NOT a circle / ellipse:

        R(phi, t) = R_lat                                  if cos(d_phi) = 0
                  = R_lat + (R_front - R_lat) * cos(d_phi)^p     (front side)
                  = R_lat + (R_back  - R_lat) * (-cos(d_phi))^p  (back  side)

        R_front = R_body + R_dyn                          # full anticipation
        R_lat   = R_body + lateral_share * R_dyn          # narrow sides
        R_back  = R_body + back_share    * R_dyn          # almost no buffer behind

    where ``d_phi = phi - heading(t)`` and ``p = front_sharpness`` controls
    how pointy the teardrop is. ``heading(t)`` is taken from the Step-A
    mean trajectory at that horizon step.

The resulting per-step shape is sampled into a closed polygon ("lobe")
of ``n_lobe_points`` vertices. The polygon is the union of the Step-A 95 %
confidence ellipse (uncertainty in WHERE the worker will be) and the
forward-biased buffer (safety margin AROUND that position).

Output (consumed by Step C / D / E of the workflow)
---------------------------------------------------
- ``hard_lobes``    (T, N, 2)  -- closed-polygon teardrops, NO-GO zone
- ``soft_lobes``    (T, N, 2)  -- same shape grown by ``soft_margin``
- ``buffer_curve``  (T,)       -- R_front(t) profile (largest dimension)
- ``component_breakdown``      -- contribution of every physical component

Visualization
-------------
Combines Step A (observation + intent + particle rollout + KDE + 95 %
ellipses) with Step B (forward-biased teardrop lobes, swept safety hull,
hard NO-GO @ t=0, time-varying safety-radius curves, inflation breakdown).

Run
---
    python safety_inflation.py                 # save mp4 to outputs/
    python safety_inflation.py --preview       # open a live window
    python safety_inflation.py --workers 3 --frames 280
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
from matplotlib.patches import Circle, Patch, Polygon, Rectangle

try:
    from .human_predict import (
        GOALS,
        MAP_BOUNDS,
        OBSTACLES,
        IntentParticlePredictor,
        PredictorConfig,
        _draw_factory,
        _kde_heatmap,
        make_workers,
    )
except ImportError:                                # pragma: no cover
    from human_predict import (
        GOALS,
        MAP_BOUNDS,
        OBSTACLES,
        IntentParticlePredictor,
        PredictorConfig,
        _draw_factory,
        _kde_heatmap,
        make_workers,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SafetyInflationConfig:
    # Physical body radii.
    worker_radius: float = 0.35       # m
    amr_radius: float = 0.40          # m

    # AMR dynamics (used for latency / braking / reaction terms).
    v_amr_max: float = 0.50            # m/s   nominal fleet speed (factory pace)
    a_decel: float = 1.2              # m/s^2 comfortable / passenger-safe braking

    # Latency budget (sum used for R_latency).
    t_sense: float = 0.10             # s perception
    t_comm: float = 0.08              # s fleet-comm round trip
    t_plan: float = 0.05              # s replan overhead

    # Reaction.
    t_react: float = 0.20             # s baseline operator/control-loop reaction

    # Epistemic horizon penalty.
    forecast_alpha: float = 0.05      # m / s of lookahead (linear growth)

    # Extra ring outside the hard inflation: drop-to-slowdown zone.
    soft_margin: float = 0.45         # m

    # ---- worker-speed gating -------------------------------------------
    # The four dynamic margins (latency, braking, reaction, forecast) are
    # multiplied by alpha_h(v_h) which is a smoothstep that ramps from 0 at
    # v_h = 0 to 1 at v_h >= ``v_full_walk``. A stationary worker only keeps
    # the body radius; a worker at walking speed gets the full inflation.
    v_full_walk: float = 0.50         # m/s -- worker walking speed in factory

    # ---- forward-biased (asymmetric "teardrop") lobe -------------------
    # Walking humans push their uncertainty FORWARD; the area behind them is
    # relatively safe. The lobe radius therefore varies with the angle of
    # the boundary normal relative to the walking direction.
    #
    #   lateral_share : fraction of dynamic margin allocated to the SIDES
    #                   (perpendicular to walking direction).
    #                   1.0 = isotropic circle/ellipse (legacy behaviour);
    #                   0.0 = razor-thin pencil beam.
    #   back_share    : fraction allocated BEHIND. 0.0 = body radius only.
    #   front_sharpness: exponent shaping the front/back falloff.
    #                    Higher = pointier teardrop.
    #   n_lobe_points : polygon resolution (more = smoother teardrop).
    lateral_share: float = 0.20
    back_share: float = 0.0
    front_sharpness: float = 2.0
    n_lobe_points: int = 64


# Default frame count -- matches step_c_affected_amr.py (280 frames, dt=0.2).
DEFAULT_FRAMES_SAFETY: int = 280

# Component color palette (used for the breakdown stacked bar + legend).
COMPONENT_COLORS: dict[str, str] = dict(
    body="#4c72b0",
    latency="#dd8452",
    braking="#c44e52",
    reaction="#8172b3",
    forecast="#937860",
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SafetyInflationModel:
    """Time-varying anisotropic safety inflation around Step-A predictions."""

    def __init__(self, cfg: SafetyInflationConfig):
        self.cfg = cfg
        self.r_body = cfg.worker_radius + cfg.amr_radius
        self.r_latency = cfg.v_amr_max * (cfg.t_sense + cfg.t_comm + cfg.t_plan)
        self.r_braking = (cfg.v_amr_max ** 2) / (2.0 * cfg.a_decel)
        self.r_reaction_base = cfg.v_amr_max * cfg.t_react

    # ---- intent-aware reaction term -------------------------------------

    @staticmethod
    def normalized_entropy(p: np.ndarray) -> float:
        """Normalized Shannon entropy in [0, 1] for a discrete distribution."""
        p = np.asarray(p, dtype=float)
        p = p / (p.sum() + 1e-12)
        h = float(-np.sum(p * np.log(p + 1e-12)))
        h_max = math.log(len(p)) if len(p) > 1 else 1.0
        return float(h / max(h_max, 1e-12))

    @staticmethod
    def velocity_gain(v_human: float, v_full: float) -> float:
        """Smoothstep gate: 0 when v_human=0, 1 when v_human >= v_full."""
        x = float(np.clip(v_human / max(v_full, 1e-9), 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def reaction_radius(self, belief: np.ndarray) -> float:
        """Reaction term grows when intent is ambiguous (entropy -> 1)."""
        return self.r_reaction_base * (1.0 + self.normalized_entropy(belief))

    # ---- core formula ---------------------------------------------------

    def components(
        self,
        t_horizon: float,
        belief: np.ndarray,
        v_human: float = 0.0,
    ) -> dict[str, float]:
        """Component breakdown. Dynamic terms are scaled by velocity_gain."""
        g = self.velocity_gain(v_human, self.cfg.v_full_walk)
        return dict(
            body=self.r_body,
            latency=self.r_latency * g,
            braking=self.r_braking * g,
            reaction=self.reaction_radius(belief) * g,
            forecast=self.cfg.forecast_alpha * t_horizon * g,
        )

    def buffer_radius(
        self,
        t_horizon: float,
        belief: np.ndarray,
        v_human: float = 0.0,
    ) -> float:
        return float(sum(self.components(t_horizon, belief, v_human).values()))

    # ---- asymmetric / forward-biased lobe -------------------------------

    def lobe_radii(
        self,
        t_horizon: float,
        belief: np.ndarray,
        v_human: float = 0.0,
    ) -> dict[str, float]:
        """Front / lateral / back composite radii for the teardrop lobe."""
        c = self.components(t_horizon, belief, v_human)
        r_body = c["body"]
        r_dyn = c["latency"] + c["braking"] + c["reaction"] + c["forecast"]
        return dict(
            R_body=r_body,
            R_dyn=r_dyn,
            R_front=r_body + r_dyn,
            R_lat=r_body + self.cfg.lateral_share * r_dyn,
            R_back=r_body + self.cfg.back_share * r_dyn,
        )

    def _compose_lobe(
        self,
        ellipse: np.ndarray,
        heading_rad: float,
        R_front: float,
        R_lat: float,
        R_back: float,
    ) -> np.ndarray:
        """Sample the Step-A ellipse boundary and push each point outward by
        ``R(phi)`` where ``phi`` is the angle of the outward normal w.r.t.
        the walking direction.  Returns a (n_lobe_points, 2) closed polygon."""
        n = int(self.cfg.n_lobe_points)
        p = float(self.cfg.front_sharpness)
        cx, cy, w, h, ang_deg = (float(x) for x in ellipse)
        a = max(w * 0.5, 1e-6)
        b = max(h * 0.5, 1e-6)
        ang = math.radians(ang_deg)
        cosA, sinA = math.cos(ang), math.sin(ang)
        out = np.zeros((n, 2))
        for i in range(n):
            t = 2.0 * math.pi * i / n
            # Ellipse boundary point (rotated to world frame)
            ex = a * math.cos(t)
            ey = b * math.sin(t)
            px = cx + ex * cosA - ey * sinA
            py = cy + ex * sinA + ey * cosA
            # Outward normal in local frame -> world frame
            # For an axis-aligned ellipse, the un-normalised outward normal
            # at param t is (b cos t, a sin t).
            nlx = b * math.cos(t)
            nly = a * math.sin(t)
            nwx = nlx * cosA - nly * sinA
            nwy = nlx * sinA + nly * cosA
            nn = math.hypot(nwx, nwy)
            if nn < 1e-9:
                continue
            nwx /= nn
            nwy /= nn
            # cosine of the angle between this outward normal and the
            # walking direction: +1 in front, 0 to the side, -1 behind.
            c = nwx * math.cos(heading_rad) + nwy * math.sin(heading_rad)
            if c >= 0.0:
                r = R_lat + (R_front - R_lat) * (c ** p)
            else:
                r = R_lat + (R_back - R_lat) * ((-c) ** p)
            out[i, 0] = px + nwx * r
            out[i, 1] = py + nwy * r
        return out

    @staticmethod
    def _speed_profile(mean_traj: np.ndarray | None, T: int, dt: float,
                       fallback: float) -> np.ndarray:
        """Per-horizon-step predicted speed in m/s, derived from mean_traj."""
        if mean_traj is None or len(mean_traj) < 2:
            return np.full(T, fallback)
        v = np.zeros(T)
        last_idx = len(mean_traj) - 1
        for t in range(T):
            i = t if t < last_idx else last_idx - 1
            v[t] = float(np.linalg.norm(mean_traj[i + 1] - mean_traj[i]) / dt)
        return v

    @staticmethod
    def _heading_profile(
        mean_traj: np.ndarray | None,
        ellipses: np.ndarray,
        T: int,
    ) -> np.ndarray:
        """Per-horizon-step walking direction [rad]. Falls back to the
        ellipse major-axis orientation when the worker is too slow."""
        headings = np.zeros(T)
        last_valid = 0.0
        L = 0 if mean_traj is None else len(mean_traj)
        for t in range(T):
            n = 0.0
            if L >= 2:
                if t == 0:
                    d = mean_traj[1] - mean_traj[0]
                elif t < L - 1:
                    d = mean_traj[t + 1] - mean_traj[t - 1]
                else:
                    d = mean_traj[L - 1] - mean_traj[L - 2]
                n = float(np.linalg.norm(d))
            if n < 1e-4:
                # Stationary -- buffer collapses anyway; keep last direction
                # so the lobe doesn't flip orientation randomly.
                if t > 0:
                    headings[t] = last_valid
                else:
                    headings[t] = math.radians(float(ellipses[t][4]))
            else:
                headings[t] = math.atan2(float(d[1]), float(d[0]))
                last_valid = headings[t]
        return headings

    def inflate_all(
        self,
        ellipses: np.ndarray,
        dt: float,
        belief: np.ndarray,
        mean_traj: np.ndarray | None = None,
        v_current: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build hard + soft asymmetric lobes for every horizon step.

        Returns
        -------
        hard_lobes : (T, N, 2) array of teardrop polygons (NO-GO).
        soft_lobes : (T, N, 2) array of grown polygons (slowdown ring).
        front_curve: (T,) max forward radius R_front(t) for the legacy plots.
        """
        T = len(ellipses)
        N = int(self.cfg.n_lobe_points)
        v_profile = self._speed_profile(mean_traj, T, dt, fallback=v_current)
        h_profile = self._heading_profile(mean_traj, ellipses, T)
        hard = np.empty((T, N, 2))
        soft = np.empty((T, N, 2))
        curve = np.empty(T)
        sm = float(self.cfg.soft_margin)
        for t in range(T):
            t_h = (t + 1) * dt
            r = self.lobe_radii(t_h, belief, v_profile[t])
            curve[t] = r["R_front"]
            hard[t] = self._compose_lobe(
                ellipses[t], h_profile[t],
                r["R_front"], r["R_lat"], r["R_back"],
            )
            # Soft tube: same shape grown by a constant margin in every
            # direction (front/back/sides all gain ``soft_margin``).
            soft[t] = self._compose_lobe(
                ellipses[t], h_profile[t],
                r["R_front"] + sm,
                r["R_lat"]   + sm,
                r["R_back"]  + sm,
            )
        return hard, soft, curve


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    rgb = mcolors.to_rgb(color)
    return (*rgb, float(np.clip(alpha, 0.0, 1.0)))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone-chain convex hull. Returns ordered CCW vertices."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def safety_tube_polygon(lobes: np.ndarray) -> np.ndarray:
    """Convex hull enclosing every per-step lobe across the horizon.

    The input is expected to be a stack of closed polygons of shape
    ``(T, N, 2)``. The output is the convex hull of all polygon vertices --
    a single smooth "swept volume" representing the entire 5 s safety
    reservation for one worker (AMR must stay outside).
    """
    lobes = np.asarray(lobes)
    if lobes.ndim == 3:
        pts = lobes.reshape(-1, 2)
    else:
        pts = lobes.reshape(-1, 2)
    return _convex_hull(pts)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (works for non-convex polygons too)."""
    x = float(point[0])
    y = float(point[1])
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def polygon_signed_radius(point: np.ndarray, polygon: np.ndarray,
                          centroid: np.ndarray | None = None) -> float:
    """Cheap "how far inside / outside" metric for a polygon.

    Returns the radial ratio ``dist(point, centroid) / mean_polygon_radius``.
    Values < 1 are approximately inside, > 1 outside. Used in place of the
    exact distance-to-polygon to drive the Step-C ``margin`` cosmetic field.
    """
    poly = np.asarray(polygon)
    if centroid is None:
        centroid = poly.mean(axis=0)
    d = float(np.linalg.norm(np.asarray(point) - centroid))
    r_mean = float(np.linalg.norm(poly - centroid, axis=1).mean())
    return d / max(r_mean, 1e-9)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def build_safety_animation(
    output_path: Path | None,
    num_frames: int = DEFAULT_FRAMES_SAFETY,
    num_workers: int = 2,
    preview: bool = False,
    seed: int = 7,
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

    horizon_times = np.arange(1, cfg.horizon_steps + 1) * cfg.dt

    # ----- figure layout -----
    fig = plt.figure(figsize=(15.8, 10.2), dpi=120)
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[3.0, 1.05],
        height_ratios=[11.5, 4.0, 0.9],
        wspace=0.20, hspace=0.34,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_intent = fig.add_subplot(gs[0, 1])
    ax_rad = fig.add_subplot(gs[1, 0])
    ax_break = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")

    _draw_factory(ax)
    ax.set_title(
        "Steps A + B: worker trajectory prediction + safety inflation",
        fontsize=13, pad=10,
    )

    # ----- KDE heatmap (Step A) -----
    grid_x = np.linspace(MAP_BOUNDS[0], MAP_BOUNDS[1], 80)
    grid_y = np.linspace(MAP_BOUNDS[2], MAP_BOUNDS[3], 48)
    heat_im = ax.imshow(
        np.zeros((len(grid_y), len(grid_x))),
        extent=(MAP_BOUNDS[0], MAP_BOUNDS[1], MAP_BOUNDS[2], MAP_BOUNDS[3]),
        origin="lower", cmap="Oranges", alpha=0.40, vmin=0.0, vmax=1.0,
        zorder=1,
    )

    # ----- per-worker visual handles -----
    # Re-designed: instead of stacking many ellipses, we draw exactly THREE
    # shapes per worker for the safety inflation:
    #   * one filled "hard" swept tube (convex hull of all inflated ellipses)
    #   * one dotted "soft" outer tube (hard tube grown by ``soft_margin``)
    #   * one red NO-GO ellipse at t=0  (current forbidden region)
    # This collapses the previous 30+ ellipse stack into 3 clean shapes.
    w_artists = []
    placeholder = np.array([[0.0, 0.0], [0.0, 1e-3], [1e-3, 0.0]])
    for w in workers:
        hist_line, = ax.plot([], [], "-", lw=2.0, color=w["color"], zorder=5)
        dot = Circle((0, 0), 0.22, facecolor=w["color"],
                     edgecolor="white", lw=1.0, zorder=6)
        ax.add_patch(dot)
        # Always-on name tag pinned to the worker's current position.
        tag = ax.text(0, 0, w["name"], fontsize=8.5, color="white",
                      fontweight="bold", ha="center", va="bottom",
                      bbox=dict(facecolor=w["color"], edgecolor="white",
                                lw=0.6, pad=2, boxstyle="round,pad=0.18"),
                      zorder=7)
        mean_line, = ax.plot([], [], "-", lw=2.2, color=w["color"],
                             alpha=0.85, zorder=5)
        mode_lines = [
            ax.plot([], [], "--", lw=1.1, color=w["color"], alpha=0.0, zorder=3)[0]
            for _ in range(len(GOALS))
        ]
        # Soft slowdown tube (outer, dotted, no fill).
        soft_tube = Polygon(placeholder, closed=True,
                            facecolor="none",
                            edgecolor=_rgba(w["color"], 0.55),
                            linestyle=":", lw=1.2, zorder=2)
        ax.add_patch(soft_tube)
        # Hard safety tube (inner, filled, solid edge).
        hard_tube = Polygon(placeholder, closed=True,
                            facecolor=_rgba(w["color"], 0.18),
                            edgecolor=_rgba(w["color"], 0.75),
                            linestyle="-", lw=1.6, zorder=2)
        ax.add_patch(hard_tube)
        # Hard NO-GO contour at t=0 -- now a forward-biased TEARDROP polygon
        # rather than a symmetric ellipse, so the shape of the immediate
        # forbidden zone matches the underlying asymmetric model.
        hard_now = Polygon(placeholder, closed=True,
                           facecolor="none", edgecolor="#cc0033",
                           linestyle="-", lw=2.4, alpha=0.0, zorder=4)
        ax.add_patch(hard_now)
        # Tiny arrow showing the predicted walking direction so the user
        # immediately reads which way the teardrop is pointing.
        heading_arrow = ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=w["color"],
                            lw=1.6, alpha=0.0),
            zorder=6,
        )
        w_artists.append(dict(
            hist=hist_line, dot=dot, tag=tag, mean=mean_line, modes=mode_lines,
            soft_tube=soft_tube, hard_tube=hard_tube, hard=hard_now,
            heading=heading_arrow,
        ))

    info_text = ax.text(
        MAP_BOUNDS[0] + 0.4, MAP_BOUNDS[3] - 0.3, "",
        fontsize=9, color="#222",
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        zorder=8, va="top", ha="left",
    )

    # ----- intent probability side panel -----
    ax_intent.set_title("Intent probabilities", fontsize=11, pad=8)
    ax_intent.set_xlim(0.0, 1.0)
    ax_intent.set_ylim(0.0, 1.0)
    ax_intent.set_xticks([0.0, 0.5, 1.0])
    ax_intent.set_xlabel("P(goal | history)")
    ax_intent.set_yticks([])
    ax_intent.set_facecolor("#fafafa")
    for spine in ("top", "right"):
        ax_intent.spines[spine].set_visible(False)

    G = len(GOALS)
    W = len(workers)
    margin_top = 0.04
    margin_bottom = 0.14
    gap_between_workers = 0.04
    usable = 1.0 - margin_top - margin_bottom - gap_between_workers * max(W - 1, 0)
    block_height = usable / W
    bar_height = block_height / (G + 1) * 0.85
    header_height = block_height - G * bar_height
    label_left = 0.05
    bar_left = 0.20
    bar_full_width = 0.66
    bar_text_x = bar_left + bar_full_width + 0.02

    bar_groups = []
    entropy_texts = []
    for wi, w in enumerate(workers):
        block_top = 1.0 - margin_top - wi * (block_height + gap_between_workers)
        ax_intent.text(
            label_left, block_top - header_height * 0.50, w["name"],
            fontsize=10.5, color=w["color"], fontweight="bold",
            transform=ax_intent.transAxes, va="center",
        )
        ent_txt = ax_intent.text(
            0.95, block_top - header_height * 0.50, "",
            fontsize=8.5, color="#555",
            transform=ax_intent.transAxes, va="center", ha="right",
        )
        entropy_texts.append(ent_txt)
        bars = []
        for gi in range(G):
            row_top = block_top - header_height - gi * bar_height
            y = row_top - bar_height * 0.9
            rect = Rectangle(
                (bar_left, y), 0.0, bar_height * 0.78,
                facecolor=w["color"], alpha=0.7,
                transform=ax_intent.transAxes,
            )
            ax_intent.add_patch(rect)
            ax_intent.text(
                label_left, y + bar_height * 0.39, f"G{gi+1}",
                fontsize=8.5, color="#444",
                transform=ax_intent.transAxes, va="center",
            )
            txt = ax_intent.text(
                bar_text_x, y + bar_height * 0.39, "0.00",
                fontsize=8.5, color="#222",
                transform=ax_intent.transAxes, va="center", ha="left",
            )
            bars.append((rect, txt))
        bar_groups.append(bars)

    # ----- safety-radius vs horizon panel -----
    ax_rad.set_facecolor("#fafafa")
    ax_rad.set_xlim(0.0, horizon_times[-1])
    ax_rad.set_xlabel("Lookahead horizon  t  [s]")
    ax_rad.set_ylabel("Effective safety radius [m]")
    ax_rad.set_title("Step-B safety budget along the 5 s horizon", fontsize=11, pad=6)
    ax_rad.grid(True, alpha=0.3)

    baseline_belief = np.full(G, 1.0 / G)
    baseline_curve = np.array([
        safety.buffer_radius(t, baseline_belief, v_human=safety_cfg.v_full_walk)
        for t in horizon_times
    ])
    ax_rad.plot(horizon_times, baseline_curve, "--", color="#444",
                lw=1.4,
                label="R_buffer(t) @ uniform intent, walking speed (reference)")

    rad_lines = []
    for w in workers:
        line, = ax_rad.plot([], [], "-", lw=2.2, color=w["color"],
                            label=f"{w['name']} effective R(t)")
        rad_lines.append(line)
    # Threshold bands: green safe (R_buffer-only) | amber slowdown | red reroute.
    r_base_uniform = baseline_curve[0]
    ax_rad.axhspan(0.0, r_base_uniform, facecolor="#d9ead3", alpha=0.45)
    ax_rad.axhspan(r_base_uniform, r_base_uniform + 1.6,
                   facecolor="#fff2cc", alpha=0.45)
    ax_rad.axhspan(r_base_uniform + 1.6, 12.0,
                   facecolor="#f4cccc", alpha=0.35)
    ax_rad.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.92)
    ax_rad.set_ylim(0.0, 7.5)

    # ----- inflation breakdown panel -----
    ax_break.set_facecolor("#fafafa")
    ax_break.set_title("Inflation breakdown @ t = horizon", fontsize=11, pad=6)
    ax_break.set_xlim(0.0, 4.5)
    ax_break.set_ylim(-0.5, 0.5)
    ax_break.set_yticks([])
    ax_break.set_xlabel("Radius contribution [m]")
    for spine in ("top", "right"):
        ax_break.spines[spine].set_visible(False)

    component_order = ["body", "latency", "braking", "reaction", "forecast"]
    break_rects, break_labels = [], []
    for name in component_order:
        rect = Rectangle((0.0, -0.20), 0.0, 0.40,
                         facecolor=COMPONENT_COLORS[name],
                         alpha=0.92, edgecolor="white", lw=1.0)
        ax_break.add_patch(rect)
        lbl = ax_break.text(0.0, 0.0, "", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold",
                            clip_on=False)
        break_rects.append(rect)
        break_labels.append(lbl)

    ax_break.legend(
        handles=[Patch(facecolor=COMPONENT_COLORS[n], label=n) for n in component_order],
        loc="lower right", fontsize=7.5, ncol=3, framealpha=0.92,
    )
    break_subtitle = ax_break.text(
        0.02, 0.92, "", transform=ax_break.transAxes,
        ha="left", va="top", fontsize=8.5, color="#444",
    )

    # ----- shared bottom legend -----
    legend_handles = [
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#6a3d9a", markeredgecolor="white",
                   markersize=12, linestyle="None"),
        Rectangle((0, 0), 1, 1, facecolor="#fdae6b", alpha=0.7),
        plt.Line2D([0], [0], linestyle="--", color="#888", lw=1.4),
        Patch(facecolor=_rgba("#888888", 0.20),
              edgecolor="#888", linewidth=1.6),
        plt.Line2D([0], [0], linestyle=":", color="#888", lw=1.6),
        plt.Line2D([0], [0], color="#cc0033", lw=2.4),
        plt.Line2D([0], [0], color="#444", lw=1.6,
                   marker=">", markersize=8, markevery=[1]),
    ]
    legend_labels = [
        "Candidate goals",
        "Step A: 5 s occupancy KDE",
        "Step A: per-goal mode trajectory",
        "Step B: 5 s safety tube (swept hard zone, teardrop hull)",
        "Step B: soft slowdown tube",
        "Step B: forward-biased NO-GO teardrop @ t=0",
        "Worker walking direction (drives the teardrop)",
    ]
    for w in workers:
        legend_handles.append(plt.Line2D([0], [0], color=w["color"], lw=2.4))
        legend_labels.append(w["name"])
    ax_leg.legend(legend_handles, legend_labels, loc="center", ncol=4,
                  frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=9)

    # ----- per-frame update -----
    def update(frame: int):
        heat = np.zeros((len(grid_y), len(grid_x)))

        worker_outs = []
        for w, predictor in zip(workers, predictors):
            obs_start = max(0, frame - 8)
            obs = w["truth"][obs_start: frame + 1]
            if len(obs) < 2:
                obs = w["truth"][:2]
            worker_outs.append(predictor.rollout(obs))

        ent_norms = [
            SafetyInflationModel.normalized_entropy(o["belief"])
            for o in worker_outs
        ]
        focus_wi = int(np.argmax(ent_norms))

        for wi, (w, out, art) in enumerate(zip(workers, worker_outs, w_artists)):
            belief = out["belief"]
            particles = out["particles"]
            mean_traj = out["mean"]
            mode_traj = out["mode_traj"]
            ellipses = out["ellipses"]

            # Per-step worker speed AND heading are taken from the Step-A
            # mean trajectory: buffer collapses to R_body when the worker
            # stops, and points forward when they are walking.
            hard_lobes, soft_lobes, buffer_curve = safety.inflate_all(
                ellipses, cfg.dt, belief, mean_traj=mean_traj,
            )

            # ---- Step A artists ----
            art["hist"].set_data(w["truth"][: frame + 1, 0],
                                 w["truth"][: frame + 1, 1])
            art["dot"].center = tuple(w["truth"][frame])
            art["tag"].set_position((float(w["truth"][frame, 0]),
                                     float(w["truth"][frame, 1]) + 0.42))
            art["mean"].set_data(mean_traj[:, 0], mean_traj[:, 1])
            for gi, line in enumerate(art["modes"]):
                line.set_data(mode_traj[gi, :, 0], mode_traj[gi, :, 1])
                line.set_alpha(float(np.clip(belief[gi] * 1.3, 0.05, 0.95)))
                line.set_linewidth(1.0 + 2.2 * float(belief[gi]))

            # ---- Step B swept safety tubes (convex hull of all teardrops) ----
            art["hard_tube"].set_xy(safety_tube_polygon(hard_lobes))
            art["soft_tube"].set_xy(safety_tube_polygon(soft_lobes))

            # ---- hard NO-GO teardrop at t=0 ----
            art["hard"].set_xy(hard_lobes[0])
            art["hard"].set_alpha(0.90)

            # ---- heading arrow showing which way the teardrop points ------
            if len(mean_traj) >= 2:
                d = mean_traj[1] - mean_traj[0]
                nrm = float(np.linalg.norm(d))
                if nrm > 1e-4:
                    head_dir = d / nrm
                    tail = tuple(w["truth"][frame])
                    tip = (tail[0] + float(head_dir[0]) * 0.85,
                           tail[1] + float(head_dir[1]) * 0.85)
                    art["heading"].set_position(tail)
                    art["heading"].xy = tip
                    art["heading"].arrow_patch.set_alpha(0.85)
                else:
                    art["heading"].arrow_patch.set_alpha(0.0)
            else:
                art["heading"].arrow_patch.set_alpha(0.0)

            # ---- KDE heatmap ----
            heat += _kde_heatmap(particles, grid_x, grid_y)

            # ---- intent bars ----
            for gi, (rect, txt) in enumerate(bar_groups[wi]):
                rect.set_width(bar_full_width * float(belief[gi]))
                txt.set_text(f"{belief[gi]:.2f}")
            entropy_texts[wi].set_text(f"H/Hmax={ent_norms[wi]:.2f}")

            # ---- safety-radius curve ----
            semi_max = np.maximum(ellipses[:, 2], ellipses[:, 3]) / 2.0
            r_total = semi_max + buffer_curve
            rad_lines[wi].set_data(horizon_times, r_total)

        if heat.max() > 1e-6:
            heat = heat / heat.max()
        heat_im.set_data(heat)

        # ---- inflation-breakdown bar (focus worker) ----
        focus_belief = worker_outs[focus_wi]["belief"]
        focus_mean = worker_outs[focus_wi]["mean"]
        if len(focus_mean) >= 2:
            v_at_T = float(np.linalg.norm(focus_mean[-1] - focus_mean[-2]) / cfg.dt)
        else:
            v_at_T = 0.0
        alpha_h = safety.velocity_gain(v_at_T, safety_cfg.v_full_walk)
        comps = safety.components(horizon_times[-1], focus_belief, v_human=v_at_T)
        left = 0.0
        for ki, name in enumerate(component_order):
            val = comps[name]
            rect = break_rects[ki]
            rect.set_xy((left, -0.20))
            rect.set_width(val)
            lbl = break_labels[ki]
            lbl.set_position((left + val * 0.5, 0.0))
            lbl.set_text(f"{val:.2f}")
            left += val
        break_subtitle.set_text(
            f"focus: {workers[focus_wi]['name']}   "
            f"H/Hmax={ent_norms[focus_wi]:.2f}   "
            f"v_h(T)={v_at_T:.2f} m/s   "
            f"alpha_h={alpha_h:.2f}   "
            f"R_buffer(T)={left:.2f} m"
        )

        r_max = safety.buffer_radius(
            horizon_times[-1], baseline_belief, v_human=safety_cfg.v_full_walk,
        )
        info_text.set_text(
            f"frame {frame+1}/{num_frames}    "
            f"horizon {cfg.horizon_steps * cfg.dt:.1f} s    "
            f"AMR v_max={safety_cfg.v_amr_max:.1f} m/s    "
            f"R_buffer range  [{safety.r_body:.2f}  ->  {r_max:.2f}] m  "
            f"(body-only  ->  walking, uniform intent)\n"
            "Step A:  recursive-Bayes intent  +  social-force rollout  +  95% ellipses  +  KDE\n"
            "Step B:  R_body + alpha_h(v_human) * (latency + braking + reaction*(1+H) + forecast*t)"
        )

        if not preview and (frame % 5 == 0 or frame == num_frames - 1):
            print(f"  rendering frame {frame+1}/{num_frames}")

        artists = [heat_im, info_text, break_subtitle]
        for art in w_artists:
            artists += [art["hist"], art["dot"], art["tag"], art["mean"]]
            artists += art["modes"]
            artists += [art["soft_tube"], art["hard_tube"], art["hard"]]
        for bars in bar_groups:
            for rect, txt in bars:
                artists += [rect, txt]
        artists += entropy_texts
        artists += rad_lines
        artists += break_rects + break_labels
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
    return Path(__file__).resolve().parents[2] / "outputs" / "safety_inflation_demo.mp4"


def main(argv: list[str] | None = None) -> None:
    pa = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pa.add_argument("--output", type=str, default=str(_default_output()),
                    help="output path for the .mp4 (or .gif fallback)")
    pa.add_argument("--frames", type=int, default=DEFAULT_FRAMES_SAFETY)
    pa.add_argument("--workers", type=int, default=2, choices=[1, 2, 3, 4])
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
    saved = build_safety_animation(
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
