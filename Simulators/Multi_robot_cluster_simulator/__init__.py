"""Standalone multi-robot cluster local-replanning simulator (RL training).

Separate from the deployment Centralized_Local_Planner: here we randomly
generate small dense clusters (1-6 AMRs on parallel/perpendicular rails + a
crossing worker) and train an attention policy to route all AMRs back onto
their reference paths without collision."""
