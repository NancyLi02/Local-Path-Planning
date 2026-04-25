from __future__ import annotations

import numpy as np


class PurePursuitController:
    """Converts a world-frame goal point to unicycle (v, omega) commands."""

    def __init__(self, max_v: float = 1.0, max_omega: float = 1.0):
        self.max_v = max_v
        self.max_omega = max_omega

    def compute(
        self,
        rx: float,
        ry: float,
        rtheta: float,
        goal: np.ndarray,
    ) -> tuple[float, float]:
        dx, dy = goal[0] - rx, goal[1] - ry
        c, s = np.cos(rtheta), np.sin(rtheta)
        lx = c * dx + s * dy
        ly = -s * dx + c * dy

        L = max(np.hypot(lx, ly), 0.3)
        kappa = 2.0 * ly / (L * L)

        v = self.max_v * float(np.clip(L / 2.0, 0.3, 1.0))
        v *= float(np.clip(1.0 - 0.5 * abs(kappa), 0.2, 1.0))

        omega = float(np.clip(v * kappa, -self.max_omega, self.max_omega))
        return v, omega