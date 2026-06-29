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
│   └── replanning.py     (E)  # V0Replanner, ReplanConfig, SpaceTimeReservation
└── viz/                       # matplotlib animation (demo only)
    ├── render_common.py       # draw_factory, kde_heatmap, AMR body, STATUS_COLOR ...
    ├── render_prediction.py   # Step A demo
    ├── render_safety.py       # Steps A+B demo
    ├── render_affected.py     # Steps A→C demo
    ├── render_clusters.py     # Steps A→D demo
    └── render_replanning.py   # Steps A→E demo (V0 actions + Gantt)
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

### Known limitation → motivates V1

A worker walking straight into a stationary, rail-bound AMR cannot be avoided by
speed alone. **V1** will add an attention-based centralized shared policy for
joint action proposal (with richer maneuvers), kept behind this same safety shield.
