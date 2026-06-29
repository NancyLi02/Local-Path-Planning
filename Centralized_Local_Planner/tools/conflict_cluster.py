"""Step D -- conflict-cluster construction.

Couples AMRs that must be co-replanned and builds the local replanning
region (ConflictCluster, ClusterResult, ConflictClusterBuilder)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .geometry import _convex_hull, _expand_hull
from .affected_amr import AMR, ConflictResult


CLUSTER_PALETTE: list[str] = [
    "#e63946",   # red
    "#2196f3",   # blue
    "#ff9800",   # orange
    "#4caf50",   # green
    "#9c27b0",   # purple
    "#00bcd4",   # cyan
]

SINGLETON_COLOR = "#9e9e9e"   # grey -- affected but isolated, no co-planning needed

@dataclass
class ConflictCluster:
    """A group of AMRs that must be co-replanned together.

    Attributes
    ----------
    cluster_id : int
        Unique cluster index this frame (recomputed every frame).
    member_names : list[str]
        Names of member AMRs (sorted alphabetically for stable display).
    trigger_workers : list[str]
        Distinct worker names that caused at least one REPLAN/SLOWDOWN in
        this cluster.
    min_ttc : float
        Minimum TTC among all member AMRs (most urgent conflict).
    worst_status : str
        REPLAN if any member is REPLAN, else SLOWDOWN.
    positions : np.ndarray   shape (M, 2)
        Current world-frame positions of members.
    hull : np.ndarray        shape (K, 2)
        Convex hull of member positions + ghost footprint, expanded by buffer.
    bbox : tuple[float,float,float,float]
        (x_min, y_min, width, height) of the local-replanning bounding box.
    coupling_pairs : list[tuple[str,str]]
        Which member pairs are space-time-coupled (min pred-dist < threshold).
    color : str
        Display colour for this cluster.
    """
    cluster_id: int
    member_names: list[str]
    trigger_workers: list[str]
    min_ttc: float
    worst_status: str
    positions: np.ndarray
    hull: np.ndarray
    bbox: tuple[float, float, float, float]
    coupling_pairs: list[tuple[str, str]]
    color: str = SINGLETON_COLOR


@dataclass
class ClusterResult:
    """Full Step D output for one simulation frame."""
    clusters: list[ConflictCluster]
    singletons: list[str]          # affected AMRs with no spatial coupling
    # N×N matrix: min predicted distance between every active-AMR pair over
    # the 5-s horizon. Used for the coupling heat-map.
    dist_matrix: np.ndarray        # shape (N_active, N_active)
    active_names: list[str]        # row/column labels for dist_matrix


class ConflictClusterBuilder:
    """Build Step-D clusters from Step-C ConflictResult objects.

    Algorithm
    ---------
    1. Mark every AMR whose Step-C status is REPLAN or SLOWDOWN as *affected*.
    2. Compute an N×N space-time proximity matrix: for every pair (i, j) of
       active AMRs, find the minimum Euclidean distance between their
       predicted positions at each shared horizon step.
    3. Build an undirected *coupling graph*:
         - An edge (i, j) exists when:
             a) both i and j are affected, OR
             b) i is affected and the min-dist(i, j) < ``cascade_dist``
                (an unaffected AMR that would be forced to move).
           AND min-dist(i, j) < ``cluster_spatial_dist``.
    4. Find connected components of the coupling graph → clusters.
    5. Unaffected singletons and fully-clear AMRs are excluded.
    """

    def __init__(
        self,
        cluster_spatial_dist: float = 2.5,   # m: space-time coupling threshold
        cascade_dist: float = 2.0,            # m: unaffected AMR pulled into cluster
        replan_region_buffer: float = 1.8,    # m: expand cluster hull by this
    ):
        self.cluster_spatial_dist = float(cluster_spatial_dist)
        self.cascade_dist = float(cascade_dist)
        self.replan_region_buffer = float(replan_region_buffer)

    # ------------------------------------------------------------------
    def build(
        self,
        amrs: list[AMR],
        results: dict[str, ConflictResult],   # name -> ConflictResult
        frame: int,
        dt: float,
        T: int,
    ) -> ClusterResult:
        # Only consider AMRs that are active this frame.
        active = [a for a in amrs if a.is_active(frame)]
        if not active:
            return ClusterResult([], [], np.zeros((0, 0)), [])

        names = [a.name for a in active]
        N = len(active)

        # --- predicted positions: (N, T, 2) ---
        pred = np.array([a.predicted_positions(dt, T) for a in active])

        # --- N×N min-distance matrix over the 5-s horizon ---
        dist_mat = np.full((N, N), np.inf)
        for i in range(N):
            dist_mat[i, i] = 0.0
            for j in range(i + 1, N):
                d = float(np.linalg.norm(pred[i] - pred[j], axis=1).min())
                dist_mat[i, j] = d
                dist_mat[j, i] = d

        # --- affected set ---
        affected_set: set[str] = set()
        for a in active:
            r = results.get(a.name)
            if r is not None and r.status in ("REPLAN", "SLOWDOWN"):
                affected_set.add(a.name)

        # --- build coupling graph edges ---
        # edge_set: set of (i, j) index pairs with i < j
        edge_set: set[tuple[int, int]] = set()
        coupling_pairs_by_name: dict[frozenset, bool] = {}

        for i in range(N):
            for j in range(i + 1, N):
                ni, nj = names[i], names[j]
                both_affected = (ni in affected_set) and (nj in affected_set)
                one_affected = (ni in affected_set) or (nj in affected_set)
                d = dist_mat[i, j]

                if both_affected and d < self.cluster_spatial_dist:
                    edge_set.add((i, j))
                    coupling_pairs_by_name[frozenset([ni, nj])] = True
                elif one_affected and d < self.cascade_dist:
                    edge_set.add((i, j))
                    coupling_pairs_by_name[frozenset([ni, nj])] = True

        # --- connected components (union-find) ---
        parent = list(range(N))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i, j in edge_set:
            union(i, j)

        # Group indices by root.
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(N):
            groups[find(i)].append(i)

        # --- build ConflictCluster objects ---
        clusters: list[ConflictCluster] = []
        singletons: list[str] = []
        cid = 0

        for root, idxs in sorted(groups.items()):
            member_names = sorted([names[i] for i in idxs])
            group_affected = [n for n in member_names if n in affected_set]

            # Only emit a cluster if at least one member is affected.
            if not group_affected:
                continue

            if len(member_names) == 1:
                singletons.append(member_names[0])
                continue

            # Gather worst TTC and status among members.
            ttcs = []
            statuses = []
            trigger_workers: list[str] = []
            for n in member_names:
                r = results.get(n)
                if r is not None:
                    ttcs.append(r.ttc if r.ttc < float("inf") else 999.0)
                    statuses.append(r.status)
                    if r.closest_worker:
                        trigger_workers.append(r.closest_worker)

            min_ttc = min(ttcs) if ttcs else float("inf")
            worst = "REPLAN" if "REPLAN" in statuses else "SLOWDOWN"
            trigger_workers = list(dict.fromkeys(trigger_workers))  # unique, ordered

            # Positions and hull.
            member_amrs = [active[names.index(n)] for n in member_names]
            positions = np.array([a.position_at(a.progress) for a in member_amrs])

            # Include ghost footprints (predicted positions at TTC) for a
            # better-fitting hull.
            ghost_pts = [positions]
            for a in member_amrs:
                r = results.get(a.name)
                if r is not None and r.pred_positions is not None:
                    ghost_pts.append(r.pred_positions)
            all_pts = np.vstack(ghost_pts)

            if len(all_pts) >= 3:
                hull_raw = _convex_hull(all_pts)
                hull = _expand_hull(hull_raw, self.replan_region_buffer)
            else:
                # Degenerate: just a padded bounding box.
                centre = all_pts.mean(axis=0)
                r_pad = self.replan_region_buffer + 1.0
                hull = np.array([
                    centre + np.array([-r_pad, -r_pad]),
                    centre + np.array([+r_pad, -r_pad]),
                    centre + np.array([+r_pad, +r_pad]),
                    centre + np.array([-r_pad, +r_pad]),
                ])

            x_min, y_min = hull.min(axis=0)
            x_max, y_max = hull.max(axis=0)
            bbox = (float(x_min), float(y_min),
                    float(x_max - x_min), float(y_max - y_min))

            # Coupling pairs inside this cluster.
            cp = []
            for ii in range(len(member_names)):
                for jj in range(ii + 1, len(member_names)):
                    pair = frozenset([member_names[ii], member_names[jj]])
                    if pair in coupling_pairs_by_name:
                        cp.append((member_names[ii], member_names[jj]))

            color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
            clusters.append(ConflictCluster(
                cluster_id=cid,
                member_names=member_names,
                trigger_workers=trigger_workers,
                min_ttc=min_ttc,
                worst_status=worst,
                positions=positions,
                hull=hull,
                bbox=bbox,
                coupling_pairs=cp,
                color=color,
            ))
            cid += 1

        return ClusterResult(
            clusters=clusters,
            singletons=singletons,
            dist_matrix=dist_mat,
            active_names=names,
        )
