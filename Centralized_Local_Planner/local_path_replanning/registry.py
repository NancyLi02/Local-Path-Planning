"""The four local replanning methods, in one place.

Everything that has to choose a method -- the evaluation harness, the
renderer, the sweeps, ``Centralized_Local_Planner.main`` -- resolves it here,
so adding a fifth method (an MPC formulation, say) means editing this file and
nothing else.

Three of the four subclass ``common.planner_base.ShieldedReplanner`` and are
driven by ``common.runtime.LocalReplanningRuntime``. ``speed_adjusting`` is the
exception: it is the original rail planner and plugs straight into
``Centralized_Local_Planner.main.Pipeline`` instead, so it is run through
``speed_adjusting.adapter``, which reports the same metrics and feeds the same
renderer. ``runs_in_shared_runtime`` is how a caller tells the two apart.
"""
from __future__ import annotations

from pathlib import Path

# Canonical order: increasing freedom of action.
METHODS = ("stop_and_go", "speed_adjusting", "optimization_based", "learning_based")

# Short label for tables.
METHOD_LABELS = {
    "stop_and_go":        "stop-and-go",
    "speed_adjusting":    "speed adj.",
    "optimization_based": "optimization",
    "learning_based":     "learning",
}

# Full description for plot titles.
METHOD_TITLES = {
    "stop_and_go":        "stop-and-go — drive at full speed or halt",
    "speed_adjusting":    "speed adjusting — speed steps 1 / ⅔ / ⅓ / 0 on the rail",
    "optimization_based": "optimization-based — speed + lateral, lowest-cost safe candidate",
    "learning_based":     "learning-based — attention policy proposes, shield decides",
}

# The action set each method may draw from -- the variable under study.
METHOD_ACTION_SETS = {
    "stop_and_go":        "{full speed on the path, stop}",
    "speed_adjusting":    "speed x {1, 2/3, 1/3, 0}, on the rail",
    "optimization_based": "(goal_fwd, goal_lat, speed_scale), 1 m of lateral freedom",
    "learning_based":     "(goal_fwd, goal_lat, speed_scale), 1 m of lateral freedom",
}

# Methods driven by common.runtime; speed_adjusting runs in its own pipeline.
SHARED_RUNTIME_METHODS = ("stop_and_go", "optimization_based", "learning_based")

# Accepted for the commands recorded in archive/ -- the names these methods
# carried before the package was reorganised.
ALIASES = {
    "stopgo": "stop_and_go",
    "rail_v0": "speed_adjusting",
    "v0": "optimization_based",
    "v1": "learning_based",
}

DEFAULT_MODEL = "logs/local_path_replanning/learning_based/best.pt"

_REPO = Path(__file__).resolve().parents[2]


def resolve(method: str) -> str:
    """Canonical method name; accepts the pre-reorganisation aliases."""
    m = ALIASES.get(method, method)
    if m not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {list(METHODS)}")
    return m


def runs_in_shared_runtime(method: str) -> bool:
    return resolve(method) in SHARED_RUNTIME_METHODS


def make_planner(method: str, config, model: str | None = None,
                 device: str = "cpu"):
    """Build the replanner for ``method``.

    Only the shared-runtime methods can be built here -- ``speed_adjusting``
    drives its own pipeline and is run via ``speed_adjusting.adapter.run``.
    """
    method = resolve(method)

    if method == "stop_and_go":
        from .stop_and_go.planner import StopAndGoReplanner
        return StopAndGoReplanner(config)

    if method == "optimization_based":
        from .optimization_based.planner import OptimizationBasedReplanner
        return OptimizationBasedReplanner(config)

    if method == "learning_based":
        from .learning_based.planner import LearningBasedReplanner
        from .learning_based.policy import MultiAMRAttentionPolicy
        if model is None:
            # Untrained network -- useful for shape checks, not for results.
            return LearningBasedReplanner(MultiAMRAttentionPolicy(config), config,
                                          device=device)
        path = Path(model)
        if not path.is_absolute():
            path = _REPO / path
        return LearningBasedReplanner.from_checkpoint(str(path), config, device=device)

    raise ValueError(
        f"{method!r} does not run in the shared runtime; "
        f"use Centralized_Local_Planner.local_path_replanning.speed_adjusting.adapter"
    )
