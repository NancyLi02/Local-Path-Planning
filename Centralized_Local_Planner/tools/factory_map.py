"""Factory map constants: workstations, candidate goals, map bounds.

Shared by every pipeline step (Steps A-D)."""
from __future__ import annotations

import numpy as np


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
