"""Checks that conflict clusters are processed independently by the shared policy.

    python -m Centralized_Local_Planner.rl.test_cluster_isolation

Two levels of evidence:

1. Policy level -- perturbing one cluster's observations must leave every other
   cluster's actions bit-identical, which is only possible if no information
   crosses cluster rows in the attention pass.
2. Env level -- in the real simulator, the observation rows must be an exact
   partition of the live local-mode AMRs that matches ``sim.locks``, and each
   AMR's nearest-peer feature must reference a member of its OWN cluster.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .local_cluster_env import LocalClusterEnv, LocalEnvConfig, OBS_DIM
from .local_policy import LocalAttentionPolicy
from .cluster_policy import act_clusters, check_cluster_partition
from .train_local import _greedy_action

_SIZES = (3, 2, 0)          # cluster 0 = [A,B,C], cluster 1 = [D,E], row 2 empty


def _make_policy(seed: int = 0):
    torch.manual_seed(seed)
    policy = LocalAttentionPolicy(OBS_DIM, hidden=64)
    policy.eval()
    return policy


def _make_batch(sizes=_SIZES, slots: int = 6, seed: int = 1):
    rng = np.random.default_rng(seed)
    C = len(sizes)
    obs = np.zeros((C, slots, OBS_DIM), dtype=np.float32)
    mask = np.zeros((C, slots), dtype=bool)
    valid = np.zeros(C, dtype=bool)
    for c, size in enumerate(sizes):
        if size:
            valid[c] = True
        for m in range(size):
            mask[c, m] = True
            obs[c, m] = rng.normal(scale=0.5, size=OBS_DIM).astype(np.float32)
    return obs, mask, valid


# ---------------------------------------------------------------------------
# 1. policy level
# ---------------------------------------------------------------------------

def test_perturbing_one_cluster_leaves_others_identical():
    """The core isolation proof: cluster 0's action cannot depend on cluster 1."""
    policy = _make_policy()
    obs, mask, valid = _make_batch()
    base, _, _ = act_clusters(policy, obs, mask, valid, "cpu", deterministic=True)

    rng = np.random.default_rng(7)
    moved = obs.copy()
    moved[1, :2] = rng.normal(scale=3.0, size=(2, OBS_DIM)).astype(np.float32)
    after, _, _ = act_clusters(policy, moved, mask, valid, "cpu", deterministic=True)

    np.testing.assert_array_equal(
        base[0], after[0],
        err_msg="cluster 0 actions changed when cluster 1 moved -> clusters are mixing")
    assert not np.array_equal(base[1], after[1]), \
        "cluster 1 actions did not change at all; the perturbation was a no-op"

    # and symmetrically
    moved0 = obs.copy()
    moved0[0, :3] = rng.normal(scale=3.0, size=(3, OBS_DIM)).astype(np.float32)
    after0, _, _ = act_clusters(policy, moved0, mask, valid, "cpu", deterministic=True)
    np.testing.assert_array_equal(
        base[1], after0[1],
        err_msg="cluster 1 actions changed when cluster 0 moved -> clusters are mixing")
    print("  [ok] perturbing one cluster leaves the other cluster's actions identical")


def test_batched_matches_one_cluster_at_a_time():
    """Batching clusters must equal running the SAME policy once per cluster."""
    policy = _make_policy()
    obs, mask, valid = _make_batch()
    batched, _, _ = act_clusters(policy, obs, mask, valid, "cpu", deterministic=True)
    for c in range(len(_SIZES)):
        if not valid[c]:
            continue
        alone, _, _ = act_clusters(policy, obs[c:c + 1], mask[c:c + 1],
                                   valid[c:c + 1], "cpu", deterministic=True)
        np.testing.assert_allclose(
            batched[c], alone[0], atol=1e-6,
            err_msg=f"cluster {c} differs when batched vs run alone")
    print("  [ok] batched clusters match one-cluster-at-a-time results")


def test_shared_weights_only():
    """No per-cluster parameters exist: one module, one parameter set."""
    policy = _make_policy()
    n_params = sum(p.numel() for p in policy.parameters())
    obs, mask, valid = _make_batch(sizes=(3, 2, 0))
    act_clusters(policy, obs, mask, valid, "cpu", deterministic=True)
    obs2, mask2, valid2 = _make_batch(sizes=(6, 0, 0))
    act_clusters(policy, obs2, mask2, valid2, "cpu", deterministic=True)
    assert sum(p.numel() for p in policy.parameters()) == n_params, \
        "policy grew parameters between cluster layouts"
    print(f"  [ok] one shared parameter set ({n_params} params) serves every cluster")


def test_empty_rows_are_inert():
    """A padded (invalid) cluster row must not affect the real clusters."""
    policy = _make_policy()
    obs, mask, valid = _make_batch(sizes=(3, 2, 0))
    with_pad, logp_pad, val_pad = act_clusters(policy, obs, mask, valid, "cpu",
                                               deterministic=True)
    no_pad, logp, val = act_clusters(policy, obs[:2], mask[:2], valid[:2], "cpu",
                                     deterministic=True)
    np.testing.assert_allclose(with_pad[:2], no_pad, atol=1e-6)
    assert abs(logp_pad - logp) < 1e-4, (logp_pad, logp)
    assert abs(val_pad - val) < 1e-4, (val_pad, val)
    print("  [ok] padded cluster rows contribute nothing to actions, logp or value")


# ---------------------------------------------------------------------------
# 2. env level
# ---------------------------------------------------------------------------

def _check_peer_features(env, obs, names, valid, atol=1e-4):
    by_name = {a.name: a for a in env.sim.amrs}
    for c in range(env.C):
        if not valid[c]:
            continue
        for m, name in enumerate(names[c]):
            xy = by_name[name].current_xy()
            peers = [by_name[o].current_xy() for o in names[c] if o != name]
            if peers:
                nearest = min(peers, key=lambda q: float(np.linalg.norm(xy - q)))
                rp = nearest - xy
                dp = float(np.linalg.norm(rp))
            else:
                rp = np.zeros(2); dp = env._diag
            want = np.array([rp[0] / 3.0, rp[1] / 3.0, min(dp / env._diag, 1.0)],
                            dtype=np.float32)
            got = obs[c, m, 6:9]
            assert np.allclose(got, want, atol=atol), (
                f"{name}: peer feature {got} is not the nearest SAME-cluster peer {want}")


def _cross_cluster_peer_count(env, names, valid):
    """How many AMRs would have picked a peer from another cluster fleet-wide."""
    by_name = {a.name: a for a in env.sim.amrs}
    owner = {n: c for c in range(env.C) if valid[c] for n in names[c]}
    everyone = list(owner)
    leaks = 0
    for name in everyone:
        xy = by_name[name].current_xy()
        others = [o for o in everyone if o != name]
        if not others:
            continue
        nearest = min(others, key=lambda o: float(
            np.linalg.norm(xy - by_name[o].current_xy())))
        if owner[nearest] != owner[name]:
            leaks += 1
    return leaks


def test_env_rows_partition_the_locks(seeds, frames):
    max_clusters = 0
    multi_frames = 0
    leak_frames = 0
    for seed in seeds:
        env = LocalClusterEnv(LocalEnvConfig(num_frames=frames))
        obs = env.reset(seed)
        done = False
        while not done:
            names = env.cluster_names
            valid = env.cluster_valid
            mask = env.cluster_mask

            check_cluster_partition(names, valid)

            live = sorted(a.name for a in env.sim.amrs
                          if a.local_mode and not a.collided)
            flat = sorted(n for c in range(env.C) if valid[c] for n in names[c])
            assert flat == live, f"rows {flat} != live local-mode AMRs {live}"
            assert int(mask.sum()) == len(flat), "mask and names disagree"

            lock_of = {n: i for i, lk in enumerate(env.sim.locks) for n in lk.members}
            for c in range(env.C):
                if not valid[c]:
                    continue
                owners = {lock_of.get(n, -1) for n in names[c]}
                assert len(owners) == 1, \
                    f"cluster row {c} spans locks {owners}: {names[c]}"

            _check_peer_features(env, obs, names, valid)

            n_now = int(valid.sum())
            max_clusters = max(max_clusters, n_now)
            if n_now >= 2:
                multi_frames += 1
                if _cross_cluster_peer_count(env, names, valid):
                    leak_frames += 1

            obs, _, done, _ = env.step(_greedy_action(obs, mask))
    return max_clusters, multi_frames, leak_frames


def main(argv=None):
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--seeds", type=int, default=4)
    pa.add_argument("--frames", type=int, default=260)
    args = pa.parse_args(argv)

    print("policy-level isolation")
    test_perturbing_one_cluster_leaves_others_identical()
    test_batched_matches_one_cluster_at_a_time()
    test_shared_weights_only()
    test_empty_rows_are_inert()

    print(f"\nenv-level partition (seeds=0..{args.seeds - 1}, frames={args.frames})")
    max_c, multi, leaks = test_env_rows_partition_the_locks(
        range(args.seeds), args.frames)
    print(f"  [ok] every observation row mapped to exactly one lock")
    print(f"  [ok] every nearest-peer feature stayed inside its own cluster")
    print(f"  max simultaneous clusters seen: {max_c}")
    print(f"  frames with >= 2 clusters:      {multi}")
    print(f"  ...of which the OLD fleet-wide code would have used a "
          f"cross-cluster peer: {leaks}")
    if max_c < 2:
        print("  NOTE: no multi-cluster frame occurred; try more seeds/frames")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
