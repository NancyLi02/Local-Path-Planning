from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def obs_normalization_scales(cfg: dict) -> tuple[float, float, float, float, float]:
    """Scales for obs normalization: lat, lookahead pos, rel vel, map size, max_v."""
    max_h_sp = float(cfg["human_speed_range"][1])
    lat_s = max(cfg["corridor_w"] * 1.5, cfg["success_lat_thresh"], 0.5)
    pos_s = max(
        cfg["n_lookahead"] * cfg["lookahead_spacing"] + 3.0,
        cfg["corridor_len"] * 0.5,
        5.0,
    )
    vel_s = max(cfg["max_v"] + max_h_sp, 0.5)
    return lat_s, pos_s, vel_s, float(cfg["map_size"]), float(cfg["max_v"])


class ReferencePath:
    """Smooth reference path built from waypoints with arc-length queries."""

    def __init__(self, waypoints: np.ndarray, num_samples: int = 1000):
        wp = np.asarray(waypoints, dtype=np.float64)
        t = np.linspace(0, 1, len(wp))
        self._cx = CubicSpline(t, wp[:, 0])
        self._cy = CubicSpline(t, wp[:, 1])

        t_f = np.linspace(0, 1, num_samples)
        self._px = self._cx(t_f)
        self._py = self._cy(t_f)

        ds = np.sqrt(np.diff(self._px) ** 2 + np.diff(self._py) ** 2)
        self._s = np.zeros(num_samples)
        self._s[1:] = np.cumsum(ds)
        self._t = t_f
        self.total_length: float = float(self._s[-1])

    def _s2t(self, s: float) -> float:
        return float(np.interp(np.clip(s, 0, self.total_length), self._s, self._t))

    def position(self, s: float) -> np.ndarray:
        t = self._s2t(s)
        return np.array([float(self._cx(t)), float(self._cy(t))])

    def tangent(self, s: float) -> np.ndarray:
        t = self._s2t(s)
        dx, dy = float(self._cx(t, 1)), float(self._cy(t, 1))
        norm = np.hypot(dx, dy) + 1e-9
        return np.array([dx / norm, dy / norm])

    def heading(self, s: float) -> float:
        t = self.tangent(s)
        return float(np.arctan2(t[1], t[0]))

    def normal(self, s: float) -> np.ndarray:
        """Left-pointing unit normal (90° counter-clockwise from tangent)."""
        t = self.tangent(s)
        return np.array([-t[1], t[0]])

    def closest_point(
        self,
        x: float,
        y: float,
        s_hint: float | None = None,
        search_radius: float = 5.0,
    ) -> tuple[float, float, float]:
        """Find the closest point on the path.

        Returns:
            (s, signed_lateral, unsigned_dist)
            signed_lateral > 0 means left of path.
        """
        if s_hint is not None:
            mask = np.abs(self._s - s_hint) <= search_radius
            if mask.sum() < 3:
                mask = np.ones(len(self._s), dtype=bool)
        else:
            mask = np.ones(len(self._s), dtype=bool)

        dx = self._px[mask] - x
        dy = self._py[mask] - y
        d2 = dx * dx + dy * dy
        idx = np.argmin(d2)

        s_cl = float(self._s[mask][idx])
        dist = float(np.sqrt(d2[idx]))

        nrm = self.normal(s_cl)
        p = self.position(s_cl)
        signed_lat = float(np.dot(np.array([x - p[0], y - p[1]]), nrm))

        return s_cl, signed_lat, dist

    def curvature(self, s: float) -> float:
        """Signed curvature κ at arc-length s."""
        t = self._s2t(s)
        dx = float(self._cx(t, 1))
        dy = float(self._cy(t, 1))
        ddx = float(self._cx(t, 2))
        ddy = float(self._cy(t, 2))
        denom = (dx * dx + dy * dy) ** 1.5 + 1e-12
        return (dx * ddy - dy * ddx) / denom

    def max_abs_curvature(self) -> float:
        """Return the maximum absolute curvature."""
        t_arr = self._t
        dx = self._cx(t_arr, 1)
        dy = self._cy(t_arr, 1)
        ddx = self._cx(t_arr, 2)
        ddy = self._cy(t_arr, 2)
        denom = (dx ** 2 + dy ** 2) ** 1.5 + 1e-12
        kappa = np.abs(dx * ddy - dy * ddx) / denom
        return float(np.max(kappa))

    def get_all_xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self._px.copy(), self._py.copy()