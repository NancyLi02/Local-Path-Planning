"""Cluster-level interface to the SHARED ``LocalAttentionPolicy``.

Step D (``tools/conflict_cluster.py``) groups the affected AMRs into conflict
clusters that are co-replanned independently: an AMR in cluster 1 must never
attend to -- or be the nearest peer of -- an AMR in cluster 2.

Independence comes from stacking the clusters along the BATCH axis instead of
the token axis: observations are ``(C, M, D)`` (C clusters, M member slots)
rather than ``(1, N, D)``. ``nn.TransformerEncoder`` with ``batch_first=True``
attends only within a batch row, so one call over C rows is mathematically
identical to C separate calls -- while staying a single kernel launch and, most
importantly, using ONE set of weights for every cluster. No policy is ever
duplicated or specialised per cluster.

    cluster 0 [A, B, C]   -> row 0 -> same policy -> [aA, aB, aC]
    cluster 1 [D, E, pad] -> row 1 -> same policy -> [aD, aE]

Rows and slots are both padded, because cluster count and cluster size vary
frame to frame. ``member_mask`` marks real AMRs inside a row; ``cluster_valid``
marks rows that hold a cluster at all. Per-cluster log-probabilities and values
are summed over valid rows so PPO callers still see one scalar per env step.
"""
from __future__ import annotations

import numpy as np
import torch

from .local_policy import ACT_DIM

__all__ = ["ACT_DIM", "n_cluster_slots", "act_clusters", "evaluate_clusters",
           "check_cluster_partition"]


def n_cluster_slots(num_amrs: int) -> int:
    """Number of cluster rows needed for a fleet of ``num_amrs`` AMRs.

    Busy-area locks hold disjoint member sets and only ever form with at least
    two members, so a fleet can never carry more than ``num_amrs // 2``
    clusters simultaneously.
    """
    return max(1, int(num_amrs) // 2)


def _to_tensors(obs, member_mask, cluster_valid, device):
    ot = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    mt = torch.as_tensor(np.asarray(member_mask), dtype=torch.bool, device=device)
    vt = torch.as_tensor(np.asarray(cluster_valid), dtype=torch.bool, device=device)
    return ot, mt, vt


@torch.no_grad()
def act_clusters(policy, obs, member_mask, cluster_valid, device,
                 deterministic: bool = False):
    """Run one independent attention pass per cluster through ``policy``.

    obs (C, M, D) / member_mask (C, M) / cluster_valid (C,). Returns
    ``(action (C, M, ACT_DIM) ndarray, logp float, value float)`` where logp and
    value are summed over the valid clusters.
    """
    ot, mt, vt = _to_tensors(obs, member_mask, cluster_valid, device)
    action, logp, value = policy.act(ot, mt, deterministic=deterministic)
    action = action * mt.unsqueeze(-1)        # padded slots carry no action
    vf = vt.float()
    return (action.cpu().numpy(),
            float((logp * vf).sum().item()),
            float((value * vf).sum().item()))


def evaluate_clusters(policy, obs, member_mask, cluster_valid, action):
    """PPO-update counterpart of :func:`act_clusters`, batched over time.

    obs (B, C, M, D), member_mask (B, C, M), cluster_valid (B, C),
    action (B, C, M, ACT_DIM). The (B, C) leading axes are flattened so every
    cluster is its own batch row, then folded back and reduced over C.
    Returns ``(logp (B,), entropy (B,), value (B,))``.
    """
    B, C, M = member_mask.shape
    logp, ent, value = policy.evaluate(
        obs.reshape(B * C, M, obs.shape[-1]),
        member_mask.reshape(B * C, M),
        action.reshape(B * C, M, action.shape[-1]),
    )
    vf = cluster_valid.float()
    logp = (logp.reshape(B, C) * vf).sum(dim=1)
    value = (value.reshape(B, C) * vf).sum(dim=1)
    # Entropy depends only on log_std, so every row carries the same number;
    # collapse to one per timestep rather than scaling it by cluster count.
    ent = ent.reshape(B, C).mean(dim=1)
    return logp, ent, value


def check_cluster_partition(cluster_names, cluster_valid) -> None:
    """Raise if the rows are not a partition (an AMR in two clusters at once)."""
    seen: dict[str, int] = {}
    for c, valid in enumerate(cluster_valid):
        if not valid:
            if cluster_names[c]:
                raise AssertionError(f"row {c} is invalid but holds {cluster_names[c]}")
            continue
        for name in cluster_names[c]:
            if name in seen:
                raise AssertionError(
                    f"{name} appears in cluster {seen[name]} and cluster {c}")
            seen[name] = c
