"""Pure geometry helpers (numpy only, no matplotlib).

Single home for the convex-hull / polygon utilities that used to be
duplicated across safety_inflation.py and step_d_conflict_cluster.py."""
from __future__ import annotations

import numpy as np


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

def _expand_hull(hull: np.ndarray, margin: float) -> np.ndarray:
    """Grow a convex polygon outward by ``margin`` metres (offset each edge)."""
    centroid = hull.mean(axis=0)
    out = []
    for pt in hull:
        d = pt - centroid
        n = np.linalg.norm(d)
        if n < 1e-9:
            out.append(pt)
        else:
            out.append(pt + d / n * margin)
    return np.array(out)
