"""Machinery shared by every local replanning method.

    cluster_state         the data contract with the simulator (or a real fleet)
    config                every tunable constant, in one place
    observation_builder   per-AMR feature vector for the learned policy
    action_decoder        raw action -> (goal_fwd, goal_lat, speed_scale)
    trajectory_rollout    action -> short-horizon trajectory + backup candidates
    safety_shield         the four safety checks, the candidate cost, the choice
    commands              what the planner hands to the tracking controller
    planner_base          rollout -> backups -> shield -> dispatch (the loop)
    worker_cache          pre-computed Step-A/B worker tubes
    runtime               the closed loop: A-D + local replanning + execution

A new method subclasses ``planner_base.ShieldedReplanner`` and overrides
``propose_actions`` (and optionally ``candidate_actions``); it inherits the
shield, the rollout and the command dispatch unchanged.
"""
