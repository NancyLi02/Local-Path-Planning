"""Shared matplotlib rendering helpers + colour palettes."""
from __future__ import annotations

import math

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from ..tools.factory_map import MAP_BOUNDS, WORKSTATIONS, GOALS


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    rgb = mcolors.to_rgb(color)
    return (*rgb, float(np.clip(alpha, 0.0, 1.0)))

STATUS_COLOR: dict[str, str] = {
    "PENDING":   "#9e9e9e",
    "CLEAR":     "#2ca02c",
    "WATCH":     "#c0ca33",      # lime  -- soft hit, observe only
    "SLOWDOWN":  "#f9a825",      # amber -- hard hit, decelerate
    "REPLAN":    "#d62728",      # red   -- TTC < t_replan, trigger Step D
    "WAITING":   "#3949ab",
    "COLLISION": "#212121",      # charcoal -- chosen distinct from AMR-C purple
    "DONE":      "#bdbdbd",
}

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

def _amr_body_polygon(
    cx: float, cy: float, heading: float,
    length: float = 1.05, width: float = 0.62,
) -> np.ndarray:
    half_l, half_w = length / 2, width / 2
    tip = half_l + 0.22
    local = np.array([
        [-half_l, -half_w],
        [+half_l * 0.65, -half_w],
        [tip, 0.0],
        [+half_l * 0.65, +half_w],
        [-half_l, +half_w],
    ])
    ca, sa = math.cos(heading), math.sin(heading)
    rot = np.array([[ca, -sa], [sa, ca]])
    return local @ rot.T + np.array([cx, cy])


def _amr_tag_position(cx: float, cy: float, heading: float) -> tuple[float, float]:
    """Place the name badge just behind the AMR body, offset to the left."""
    # Rear of body + slight lateral offset so the label clears the nose.
    tx = cx - 0.50 * math.cos(heading) - 0.30 * math.sin(heading)
    ty = cy - 0.50 * math.sin(heading) + 0.30 * math.cos(heading)
    return tx, ty


def _ghost_border_color(hard: bool, soft: bool, t_value: float,
                        t_replan_eff: float, alpha_amr: float = 1.0) -> str:
    """Per-ghost border color reflecting the planner action implied at that t.

    A near-stationary AMR (``alpha_amr ~= 0``) cannot replan or slow down, so
    every hit (hard or soft) collapses to WATCH for consistency with the
    overall status and gantt cells.
    """
    if alpha_amr < 0.05:
        return STATUS_COLOR["WATCH"] if (hard or soft) else STATUS_COLOR["CLEAR"]
    if hard:
        return STATUS_COLOR["REPLAN"] if t_value < t_replan_eff else STATUS_COLOR["SLOWDOWN"]
    if soft:
        return STATUS_COLOR["WATCH"]
    return STATUS_COLOR["CLEAR"]
