"""Learning-based: an attention policy proposes the joint action.

    policy.py    shared encoder -> self-attention over the cluster -> shared
                 actor head + centralized critic (permutation-equivariant, so
                 it handles a variable number of AMRs)
    planner.py   the replanner that runs it behind the shared safety shield
    env.py       cluster-level RL environment (one episode = one conflict cluster)
    train.py     behaviour cloning warm-start + curriculum PPO
"""
from .planner import LearningBasedReplanner
from .policy import MultiAMRAttentionPolicy
__all__ = ["LearningBasedReplanner", "MultiAMRAttentionPolicy"]
