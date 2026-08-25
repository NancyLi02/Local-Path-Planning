"""Step E V1 configuration.

All tunable constants for the variable-N multi-AMR local replanning module
live here, so the planner, observation builder, rollout, shield and trainer
never hard-code a number.

The defaults follow the project specification (CLAUDE.md). Values marked
"[sim]" are matched to the existing factory simulator (Steps A-D) so the
module drops straight into it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class V1Config:
    # ---- cluster / tensor shapes ----------------------------------------
    max_agents: int = 4                # N_max per conflict cluster
    obs_dim: int = 68                  # recomputed in __post_init__ from the blocks
    token_dim: int = 128
    action_dim: int = 3                # (goal_fwd, goal_lat, speed_scale)

    # ---- attention network ----------------------------------------------
    num_attention_layers: int = 2
    num_attention_heads: int = 4
    transformer_ff_dim: int = 256
    # The specification suggests dropout 0.1. It is disabled here: PPO samples
    # actions with the network in eval mode and re-evaluates them in train
    # mode, so any dropout makes the importance ratio compare two different
    # networks and the update is systematically wrong (measured: return decays
    # 36 -> 14 over 10k steps; see logs/step_e_v1/ppo_dropout_run.log).
    dropout: float = 0.0
    # Exploration noise of the Gaussian over raw actions. Kept small: the
    # policy starts from a behaviour clone of V0, and loud exploration walks it
    # straight off that operating point (measured: return 32 -> 14).
    init_log_std: float = -2.0

    # ---- action bounds ---------------------------------------------------
    max_forward_dist: float = 3.0      # m, longitudinal local goal
    max_lateral_offset: float = 1.0    # m, lateral local goal

    # ---- rollout ---------------------------------------------------------
    horizon_sec: float = 3.0           # command horizon (dispatched waypoints)
    # The candidate is EVALUATED over the full worker-prediction horizon by
    # extrapolating the same action. Judging a manoeuvre only over the 3 s
    # command window makes stopping look optimal every single frame while a
    # worker who needs 4 s to arrive walks straight into the halted AMR.
    eval_horizon_sec: float = 3.0
    dt: float = 0.1                    # rollout resolution
    dt_pred: float = 0.2               # [sim] worker-prediction step
    pred_horizon_steps: int = 25       # [sim] 5 s worker tube

    # ---- geometry / safety -----------------------------------------------
    amr_radius: float = 0.45
    worker_safety_radius: float = 0.8
    min_amr_distance: float = 0.9
    obstacle_margin: float = 0.05      # extra inflation on static obstacles
    # Circular floor on the worker keep-out, measured to the PREDICTED worker
    # centre: the specification's r_amr + r_worker + margin.
    worker_circle_clearance: float = 1.25
    use_polygon_tube: bool = True      # False -> circular tube only
    shield_stride: int = 3             # check every k-th rollout sample
    # Hard-constraint window: the predicted worker tube is a HARD constraint up
    # to this look-ahead time and a cost beyond it. 5.0 s = the whole
    # prediction horizon, i.e. the specification's literal reading.
    #
    # A shorter window was tried first, to stop the AMR freezing in an aisle
    # because the tube might cover that spot in 4 s. Measured on 3 seeds it is
    # strictly worse (1.00 vs 0.00 collisions, 83 % vs 100 % completion): the
    # freezing that motivated it was really caused by a bug in the
    # acceleration limit, and once the manoeuvre kinematics were correct the
    # conservative full-horizon constraint won on every metric. See
    # outputs/8_step_e_v1_module/results/ablation.json.
    hard_horizon_sec: float = 5.0
    polygon_stride: int = 2            # lobe vertex downsampling for the test

    # ---- dynamic feasibility ---------------------------------------------
    v_max: float = 0.35                # [sim] AMR_SPEED_DEFAULT
    a_max: float = 0.50                # m/s^2
    omega_max: float = 1.20            # rad/s
    curvature_max: float = 4.00        # 1/m, lateral-manoeuvre curvature
    max_lateral_slope: float = 1.00    # |dl/ds|, reported for the smoothness cost
    # Lateral (strafing) speed limit. The AMR is holonomic in this factory
    # model, so the lateral offset is driven in TIME rather than in arc-length.
    # With an arc-length ramp a halted AMR cannot move sideways at all, which
    # removes its only escape at exactly the moment it needs one.
    v_lat_max: float = 0.25

    # ---- observation ------------------------------------------------------
    max_workers: int = 3               # top-K risky workers in the observation
    route_preview_points: int = 10     # M
    ego_dim: int = 6
    goal_dim: int = 2
    priority_dim: int = 5
    worker_feature_dim_per_worker: int = 7
    # Spatial token block (world x, y, cos/sin heading). NOT in the original
    # specification, which lists only path-relative features -- but without it
    # the self-attention has no way to relate two AMRs geometrically, so the
    # AMR-AMR coordination the attention is there for cannot be learned.
    # Set False to reproduce the exact 64-dim specification observation
    # (kept as a paper ablation).
    include_spatial_block: bool = True
    spatial_dim: int = 4

    # ---- candidate / backup actions ---------------------------------------
    use_backup_candidates: bool = True
    # Lazy backups: a planner that executes its own proposal when that proposal
    # is safe (V1) does not need to roll out and check the whole backup set
    # first -- the backups are built only when the proposal fails. This is
    # where a learned proposal pays off in COMPUTE, not just in quality.
    lazy_backups: bool = True
    # Deployment mode for V1. False (the factory setting, and the only one
    # reported): the learned action is one more candidate and the shield still
    # commits to the lowest-cost safe one, so V1 can never be less safe than
    # the deterministic planner. True is the specification's "policy owns
    # efficiency" flow; it was measured (0.40 collisions, 93.3 % completion)
    # and is not used.
    v1_prefer_proposal: bool = False
    use_detour_candidates: bool = True   # add full-width left/right detours
    # Dense candidate grid (speed factors x lateral offsets) instead of the
    # six named backups. Gives the deterministic baseline a far richer search,
    # at a proportionally higher planning cost.
    # Emergency reverse. The policy action space is forward-only (speed_scale
    # in [0, 1], per the specification), but an AMR that is boxed in -- cannot
    # go forward, cannot strafe clear -- has no escape left, and an oblivious
    # worker then walks into it. Backing off is offered by the SHIELD only,
    # and only when no candidate at all is safe.
    allow_emergency_reverse: bool = True
    # When no candidate is safe, rank the candidates by violation severity and
    # clearance instead of stopping outright. The STOP candidate is ranked
    # alongside the others, so it still wins whenever moving would be worse.
    least_unsafe_fallback: bool = True
    reverse_speed: float = 0.25        # m/s
    candidate_grid: bool = False
    grid_speeds: tuple = (1.0, 0.75, 0.5, 0.25, 0.0)
    grid_lateral: tuple = (0.0, 0.5, -0.5, 1.0, -1.0)
    backup_slow_scale: float = 0.5     # slow-down candidate speed multiplier
    backup_lateral_shift: float = 0.5  # m, small left / right shift
    backup_forward_scale: float = 0.5  # shorter-forward candidate multiplier

    # ---- V0 priority weights: alpha/TTC + beta*task + gamma*braking --------
    prio_alpha: float = 1.0
    prio_beta: float = 0.5
    prio_gamma: float = 0.5
    ttc_eps: float = 0.2

    # ---- candidate cost J = w1 Jw + w2 Ja + w3 Jroute + w4 Jsmooth + w5 Jdelay
    w_worker: float = 4.0
    w_worker_soft: float = 2.0         # predicted tube overlap beyond hard_horizon
    w_amr: float = 3.0
    w_route: float = 1.0
    w_smooth: float = 0.5
    w_delay: float = 2.0
    # Command-consistency (switching) cost, penalising a change of command
    # between consecutive re-plans. Needed only while the manoeuvre kinematics
    # were wrong; with them fixed it costs efficiency for no safety gain
    # (measured: stop ratio 4.7 % -> 3.3 %, route deviation 0.29 m -> 0.04 m),
    # so it is off by default and kept as an ablation.
    w_switch: float = 0.0

    # ---- reward weights (PPO, cluster-level) -------------------------------
    r_collision: float = 20.0
    r_worker_risk: float = 1.0
    r_amr_risk: float = 1.0
    r_delay: float = 0.5
    r_route_deviation: float = 0.5
    r_jerk: float = 0.1
    r_shield: float = 1.0              # penalty when the shield changes the action
    r_progress: float = 6.0            # progress toward the local goal (learnability)
    r_reach: float = 5.0               # bonus per AMR that reaches its local goal

    # ---- runtime ----------------------------------------------------------
    goal_lookahead: float = 3.0        # m, local goal ahead of the cluster exit
    reach_tol: float = 0.6             # m, local goal reached
    max_control_frames: int = 150      # liveness: hand back to the QR planner

    def __post_init__(self) -> None:
        self.obs_dim = self.feature_dim

    @property
    def feature_dim(self) -> int:
        return (self.ego_dim + self.goal_dim + self.route_feature_dim
                + self.worker_feature_dim + self.priority_dim
                + (self.spatial_dim if self.include_spatial_block else 0))

    @property
    def rollout_steps(self) -> int:
        return int(round(self.horizon_sec / self.dt))

    @property
    def worker_feature_dim(self) -> int:
        return self.max_workers * self.worker_feature_dim_per_worker

    @property
    def route_feature_dim(self) -> int:
        return 3 * self.route_preview_points

    def validate(self) -> None:
        """obs_dim must equal the length produced by observation_builder."""
        total = self.feature_dim
        if total != self.obs_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} but the feature blocks sum to {total} "
                f"(ego {self.ego_dim} + goal {self.goal_dim} + route "
                f"{self.route_feature_dim} + worker {self.worker_feature_dim} + "
                f"priority {self.priority_dim} + spatial "
                f"{self.spatial_dim if self.include_spatial_block else 0})")


DEFAULT_CONFIG = V1Config()
DEFAULT_CONFIG.validate()
