"""Step B -- safety inflation model.

Builds forward-biased anisotropic safety lobes around Step-A predictions
(SafetyInflationModel)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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
