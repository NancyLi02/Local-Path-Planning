"""Scenario builders: scripted workers + the staggered AMR fleet."""
from __future__ import annotations

import math

import numpy as np

from .factory_map import GOALS
from .affected_amr import AMR, AMR_SPEED_DEFAULT


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

def make_stray_loader(
    num_frames: int,
    dt: float,
    collision_frame: int = 93,
    collision_xy: tuple[float, float] = (10.0, 6.5),
    walk_speed: float = 1.0,
) -> dict:
    """Untracked 'Loader' walking through the central aisle.

    The Loader is NOT seen by Step A's intent predictor and NOT wrapped in a
    Step B safety tube. From the AMR's perspective they are an unmodeled
    intruder. Their truth path is engineered to bring them through
    ``collision_xy`` at exactly ``collision_frame`` -- the time AMR-B passes
    that same point -- so the demo will reliably exhibit the physical
    AMR-human collision that the rest of the system is designed to handle.
    """
    cx, cy = collision_xy
    start = np.array([cx, cy + 3.0])                  # 3 m above the lane
    end = np.array([cx, cy - 4.5])                    # walks straight south
    dist_to_target = 3.0
    frames_to_target = int(round(dist_to_target / walk_speed / dt))
    spawn_frame = max(0, collision_frame - frames_to_target)

    truth = np.zeros((num_frames, 2))
    seg_len = float(np.linalg.norm(end - start))
    direction = (end - start) / max(seg_len, 1e-9)
    for f in range(num_frames):
        if f < spawn_frame:
            truth[f] = start
        else:
            s = walk_speed * (f - spawn_frame) * dt
            s = min(s, seg_len)
            truth[f] = start + direction * s
    return dict(
        name="Loader",
        color="#d84315",                              # burnt orange
        truth=truth,
        spawn_frame=spawn_frame,
        is_tracked=False,
    )


def make_amrs(num_amrs: int = 6) -> list[AMR]:
    """Staggered fleet with two engineered AMR-AMR crossings to showcase yielding.

    Engineered conflicts (resolved by QR-cell reservation, FCFS by
    waypoint-arrival time):
        * AMR-A (east y=5) vs AMR-D (north x=7)    -> contest cell near (7, 5)
        * AMR-A (east y=5) vs AMR-C (south x=12.5) -> contest cell near (12.5, 5)
    Whoever files the request for the contested QR cell first wins; the other
    AMR holds at its current QR waypoint (WAITING) until the cell clears.
    """
    fleet_spec = [
        # name,    color,        waypoints,                            spawn
        ("AMR-A", "#0aa6a6",
         np.array([[0.5, 5.0], [19.5, 5.0]]),                              0),
        ("AMR-B", "#2d3e8c",
         np.array([[19.5, 6.5], [0.5, 6.5]]),                               12),
        ("AMR-D", "#5d6d00",
         np.array([[7.0, 3.5], [7.0, 10.5]]),                               30),
        ("AMR-C", "#7b1fa2",
         np.array([[12.5, 10.5], [12.5, 3.5]]),                             35),
        ("AMR-E", "#c2185b",
         np.array([[0.5, 7.5], [19.5, 7.5]]),                               55),
        ("AMR-F", "#00897b",
         np.array([[18.5, 3.5], [18.5, 10.5]]),                             70),
    ]
    fleet = [
        AMR(name=n, color=c, waypoints=wp,
            commanded_speed=AMR_SPEED_DEFAULT, spawn_frame=sf)
        for (n, c, wp, sf) in fleet_spec[:num_amrs]
    ]
    # Display order = alphabetical by name (A, B, C, D, E, F).
    # AMR-AMR arbitration order is decided per-request inside
    # CentralizedPlanner by waypoint-arrival time, not by this list order.
    fleet.sort(key=lambda a: a.name)
    return fleet
