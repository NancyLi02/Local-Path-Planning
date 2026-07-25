# Centralized Local Path Planning Framework

Automated-factory AMR centralized local path planning core. Implements the
red-box pipeline of `overall_framework.png` (Steps A–D) plus the first local
replanning layer (Step E, V0).

```
A prediction ──► B safety_inflation ──► C affected_amr ──► D conflict_cluster ──► E replanning (V0)
```

Every step works on a shared factory scene: scripted **workers** walking between
workstations, and a fleet of rail-constrained **AMRs** driving QR-code reference
paths under a centralized planner.

## Layout

```
Centralized_Local_Planner/
├── main.py                    # trunk: CLI demo dispatch + Pipeline (A→B→C→D→E)
├── eval_v0.py                 # autonomous Step-E V0 evaluation + param auto-tune
├── tools/                     # pure algorithm logic (no matplotlib)
│   ├── factory_map.py         # GOALS / MAP_BOUNDS / OBSTACLES / WORKSTATIONS
│   ├── geometry.py            # convex hull, polygon tests, swept tube, hull expand
│   ├── scenario.py            # make_workers / make_amrs / make_stray_loader
│   ├── prediction.py     (A)  # PredictorConfig, IntentParticlePredictor
│   ├── safety_inflation.py(B) # SafetyInflationConfig, SafetyInflationModel
│   ├── affected_amr.py   (C)  # AMR, CentralizedPlanner, ConflictChecker
│   ├── conflict_cluster.py(D) # ConflictCluster, ClusterResult, ConflictClusterBuilder
│   └── replanning.py    (E0)  # V0Replanner, ReplanConfig, SpaceTimeReservation
├── viz/                       # matplotlib animation (demo only)
│   ├── render_common.py       # draw_factory, kde_heatmap, AMR body, STATUS_COLOR ...
│   ├── render_prediction.py   # Step A demo
│   ├── render_safety.py       # Steps A+B demo
│   ├── render_affected.py     # Steps A→C demo
│   ├── render_clusters.py     # Steps A→D demo
│   └── render_replanning.py   # Steps A→E demo (V0 actions + Gantt)
└── rl/                   (E1) # V1: attention RL behind the V0 shield
    ├── fleet_env.py           # fast env (precomputed workers + no-go table + shield)
    ├── policy.py              # AttentionPolicy (shared encoder + self-attn + Beta head)
    ├── train_v1.py            # compact PyTorch PPO (GPU)
    ├── eval_v1.py             # V0 vs V1 comparison
    └── plot_curve.py          # training-curve plot
```

Shared helpers that were previously duplicated (`_rgba`, `_convex_hull`) now
live in a single place (`viz/render_common.py` and `tools/geometry.py`).

## Run the demos

```bash
# from the workspace root (path_planning_ws)
python -m Centralized_Local_Planner.main predict   --frames 60     # Step A
python -m Centralized_Local_Planner.main safety                     # Steps A+B
python -m Centralized_Local_Planner.main affected  --amrs 6         # Steps A→C
python -m Centralized_Local_Planner.main cluster                    # Steps A→D
python -m Centralized_Local_Planner.main replan    --amrs 6         # Steps A→E (V0)
python -m Centralized_Local_Planner.main pipeline                   # render A–D demos
```

Add `--preview` to open a live window instead of writing an `.mp4`.
Rendered files go to `../outputs/`.

## Programmatic pipeline

```python
from Centralized_Local_Planner.main import Pipeline
from Centralized_Local_Planner.tools.replanning import V0Replanner, ReplanConfig

# headless A→B→C→D→E, one frame at a time
pipe = Pipeline(num_frames=360, num_workers=2, num_amrs=6,
                replanner=V0Replanner(ReplanConfig(shield_steps=25)))
for f in range(pipe.num_frames):
    out = pipe.step(f)
    # out["worker_data"] : per-worker safety tubes (Step B)
    # out["results"]     : {amr_name: ConflictResult}  (Step C status / TTC)
    # out["clusters"]    : ClusterResult (Step D groups + replanning region)
    # V0 has already set each amr.actual_speed for this frame (Step E)
```

Pass `replanner=None` for the no-shield baseline.

## Step E — V0 (TTC-priority replanning + space-time shield)

V0 is the deterministic (non-learned) safety baseline for the local replanning
layer. AMRs are **rail-constrained**, so the only local degree of freedom is the
**speed along the rail** — a "candidate trajectory" is therefore a choice of
speed. Each frame, after Steps A–D, `V0Replanner.plan()` does:

1. **TTC priority** — order active AMRs by ascending time-to-collision; the most
   urgent plans first and reserves space-time first (gets right of way). TTC is
   used only for ordering, not for the stop decision.
2. **Candidate set** — try speed factors fastest-first:
   `(1.0, 0.66, 0.33, 0.0)` × the QR-planner-allowed speed.
3. **Space-time reservation shield** — a candidate is admissible iff, over the
   shield horizon, its swept rail positions stay out of (a) every worker's
   Step-B hard no-go lobe (time-aligned) and (b) the footprints already reserved
   by higher-priority AMRs. The AMR commits to the **fastest admissible** speed
   (max throughput), reserves that footprint, and only **stops** if no non-zero
   speed is admissible.

Because slowing changes *where/when* the AMR will be, a conflict that exists at
full speed usually disappears at a slower speed — so V0 mostly **SLOWs** rather
than hard-stops; STOP is the last resort.

**Important:** Step C's `t_replan = 2.8 s` only colours the Gantt cells /
triggers Step-D clustering — it is **not** the V0 stop criterion. V0 reacts to
any hard-lobe hit within its `shield_steps` horizon (default tuned to the full
5 s). The full horizon is required: shorter shields desync AMR timing from the
QR planner and are *worse* than the baseline.

### Evaluation

```bash
python -m Centralized_Local_Planner.eval_v0 --frames 360 --seeds 5
```

Runs baseline vs V0 over multiple seeds, grid-searches `shield_steps` (the
deterministic stand-in for "training"), writes `../outputs/step_e_v0_results.json`,
and prints a comparison table. Latest result (6 AMRs / 2 workers, 5 seeds,
360 frames, tuned `shield_steps=25`):

| metric              | baseline | V0 (tuned) |   Δ      |
|---------------------|---------:|-----------:|---------:|
| worker collisions   |     2.00 |   **0.00** |  −2.00   |
| AMR–AMR collisions  |     0.00 |       0.00 |   0.00   |
| completion %        |    33.3  |   **80.0** |  +46.7   |
| path progress %     |    71.0  |   **97.3** |  +26.3   |
| min clearance [m]   |    0.54  |       0.59 |  +0.05   |
| stop ratio %        |    34.8  |   **14.2** |  −20.6   |

V0 eliminates all worker collisions while *increasing* throughput and *reducing*
stopping (collisions in the baseline permanently disable AMRs, tanking completion).

## Step E — V1 (attention RL behind the same shield)

V1 replaces V0's greedy "always full speed" with a **learned centralized
attention policy** that proposes a per-AMR desired speed factor; the proposal is
projected through the **identical V0 space-time shield**, so safety is still
guaranteed by construction and the policy only ever shapes *coordination /
efficiency*.

- **Policy** (`rl/policy.py`): a shared per-AMR MLP encoder + a small Transformer
  self-attention block (AMRs attend to each other; inactive AMRs masked) + a
  shared Beta(α,β) actor head and a centralized critic. Permutation-equivariant,
  so it handles a varying number of active AMRs.
- **Env** (`rl/fleet_env.py`): wraps Steps A–D. Workers are scripted, so their
  Step-A/B safety lobes are precomputed per seed; AMRs are rail-bound, so a
  per-seed **no-go lookup table** (`nogo[amr][frame][t][arclen]`, vectorised
  point-in-polygon, disk-cached) turns the shield check into O(steps) lookups
  (~6 ms/step). The action is given a x1.25 slack so the policy can request the
  full max-safe speed (i.e. can reproduce V0 exactly).
- **Training** (`rl/train_v1.py`): compact PyTorch PPO + GAE on GPU. The policy is
  warm-started near V0 (Beta mean ~0.8 -> full speed after slack) and learns to
  deviate only where it helps.

```bash
python -m Centralized_Local_Planner.rl.train_v1 --timesteps 120000 --train-seeds 4
python -m Centralized_Local_Planner.rl.eval_v1  --model logs/V1/v1_best.pt --seeds 5
```

Result (6 AMRs / 2 workers, 5 seeds, both behind the shield):

| metric            | V0 (greedy) | V1 (learned) |   Δ      |
|-------------------|------------:|-------------:|---------:|
| worker collisions |        0.00 |         0.00 |   0.00   |
| completion %      |       80.00 |    **83.33** |  +3.33   |
| path progress %   |       97.56 |        97.73 |  +0.17   |
| min clearance [m] |        0.57 |         0.57 |   0.00   |
| stop ratio %      |        1.57 |     **0.87** |  −0.70   |

V1 keeps the shield's zero-collision guarantee while learning fleet-level
coordination (yield to reduce mutual blocking) that lifts completion and reduces
stopping. Note V0 is already the throughput *ceiling behind the shield*
(it always drives max-safe), so gains are modest and concentrated in
coordination; larger gains need a richer action space (lateral maneuvers, not
just speed) — the natural next step.

Training curve: `outputs/step_e_v1_training_curve.png`;
metrics: `outputs/step_e_v1_results.json`.

## Step E — spatial local PATH replanning (the real local replanning)

V0/V1 above only modulate **speed along fixed rails** — no spatial detour. The
spatial layer lets a conflict cluster's AMRs temporarily **leave their rails**
and move in 2D inside a locked **busy area** (the Step-D hull), route around
workers/each other, and rejoin the reference path downstream; non-member AMRs
wait at the busy-area boundary until it clears.

- `tools/local_replanning.py` — AMR 2D local-mode + `compute_exit_s` +
  `shielded_displacement` (2D circular keep-out + peer clearance + emergency
  flee) + `RuleLocalReplanner` (greedy-to-exit + evade) + `LocalReplanSim`
  (busy-area locks, liveness timeout). Demo `viz/render_local.py`.
- `rl/` — learned version: `episodic_env.py` (per-cluster EPISODIC env with
  harvested cluster scenarios — dense signal so RL is learnable),
  `local_policy.py` (2D tanh-Gaussian attention, **separate actor/critic trunks**
  for stable BC+PPO), `train_episodic.py` (BC warm-start from greedy + critic
  warmup + PPO), `eval_local_compare.py`, `render_local_rl.py`.

```bash
python -m Centralized_Local_Planner.viz.render_local                 # rule demo
python -m Centralized_Local_Planner.rl.train_episodic --timesteps 120000
python -m Centralized_Local_Planner.rl.eval_local_compare --model logs/V1_local/episodic_best.pt
python -m Centralized_Local_Planner.rl.render_local_rl --model logs/V1_local/episodic_best.pt
```

Result (6 AMRs / 2 workers, 5 seeds, full simulator, same 2D shield):

| method (spatial)       | worker collisions | completion % | AMR-frames replanning |
|------------------------|------------------:|-------------:|----------------------:|
| **rule** (greedy+evade)|              0.00 |   **100.0**  |          193          |
| **RL** (attention)     |              1.00 |      83.3    |          233          |

Both produce genuine multi-AMR spatial path replanning (AMRs leave rails, route
around, rejoin). The hand-crafted rule is robust/near-optimal; the learned
attention policy reproduces the behaviour on its training distribution and runs
behind the identical shield, but a small imitation/transfer error in the
unforgiving full simulator costs it ~1 collision and ~1 late finish per episode.
**Finding:** on this rail-structured, near-saturated scene the analytic
replanner is hard to beat with RL; bigger RL wins need a denser/harder scene
(more AMRs, tighter aisles) where greedy routing congests.

Demos: `outputs/step_e_local_replanning_demo.mp4` (rule),
`outputs/step_e_local_rl_demo.mp4` (RL); metrics `outputs/step_e_local_results.json`.
