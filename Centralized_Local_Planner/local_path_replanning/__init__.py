"""Local path replanning -- the local layer of the centralized planner.

Steps A-D (``..tools``) predict where the workers will be, inflate that into
no-go lobes, mark which AMRs are affected, and group them into conflict
clusters. This package is what happens next: given a cluster of N <= max_agents
AMRs it decides, every frame, what each of them should actually do.

Four methods share one runtime and one safety shield, and differ only in the
action set they may draw from and in who picks the action:

    stop_and_go         drive at full speed or halt                 rule
    speed_adjusting     modulate speed along the rail (1 DOF)       rule
    optimization_based  speed + lateral offset      (3 DOF)         cost search
    learning_based      speed + lateral offset      (3 DOF)         attention policy

The design principle throughout:

    the planner proposes an efficient action,
    the safety shield decides what may execute.

Because every method is shielded by the same code, a comparison between them
measures coordination and throughput, never safety.

    from Centralized_Local_Planner.local_path_replanning import make_planner, METHODS
    planner = make_planner("learning_based", cfg, model="logs/.../best.pt")
    commands = planner.plan(cluster, worker_predictions, map_data, dt)
"""
from .registry import METHODS, METHOD_LABELS, make_planner

__all__ = ["METHODS", "METHOD_LABELS", "make_planner",
           "common", "stop_and_go", "speed_adjusting",
           "optimization_based", "learning_based"]
