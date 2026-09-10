# Centralized Local Path Planning Framework

Automated-factory AMR centralized planning core. Implements the red-box pipeline
of `../overall_framework.png` — worker prediction through conflict clustering —
and the local replanning layer that acts on it.

```
A prediction ─► B safety_inflation ─► C affected_amr ─► D conflict_cluster ─► local path replanning
```

Every step works on one shared factory scene: scripted **workers** walking
between workstations, and a fleet of rail-constrained **AMRs** driving QR-code
reference paths under a centralized planner.

## Layout

```
Centralized_Local_Planner/
├── main.py                    # trunk: Pipeline (A→B→C→D) + CLI demo dispatch
├── tools/                     # pure algorithm logic (no matplotlib)
│   ├── factory_map.py         # GOALS / MAP_BOUNDS / OBSTACLES / WORKSTATIONS
│   ├── geometry.py            # convex hull, polygon tests, swept tube, hull expand
│   ├── scenario.py            # make_workers / make_amrs / make_stray_loader
│   ├── prediction.py     (A)  # PredictorConfig, IntentParticlePredictor
│   ├── safety_inflation.py(B) # SafetyInflationConfig, SafetyInflationModel
│   ├── affected_amr.py   (C)  # AMR, CentralizedPlanner, ConflictChecker
│   └── conflict_cluster.py(D) # ConflictCluster, ClusterResult, ConflictClusterBuilder
├── viz/                       # matplotlib animation (demo only)
│   ├── render_common.py       # draw_factory, kde_heatmap, AMR body, STATUS_COLOR ...
│   ├── render_prediction.py   # Step A demo
│   ├── render_safety.py       # Steps A+B demo
│   ├── render_affected.py     # Steps A→C demo
│   └── render_clusters.py     # Steps A→D demo
└── local_path_replanning/     # the local layer -- four methods, one shield
    ├── registry.py            # the four methods; every switch resolves here
    ├── common/                # runtime, shield, rollout, observation, commands
    ├── stop_and_go/           # drive or halt
    ├── speed_adjusting/       # speed on the rail
    ├── optimization_based/    # speed + lateral, cost search   (MPC slot)
    ├── learning_based/        # speed + lateral, attention policy
    └── README.md              # ← the detail for all four lives there
```

## Run the demos

```bash
# from the workspace root (path_planning_ws)
python -m Centralized_Local_Planner.main predict   --frames 60     # Step A
python -m Centralized_Local_Planner.main safety                     # Steps A+B
python -m Centralized_Local_Planner.main affected  --amrs 6         # Steps A→C
python -m Centralized_Local_Planner.main cluster                    # Steps A→D
python -m Centralized_Local_Planner.main pipeline                   # render A–D
```

The local replanning demo takes the **method** as a parameter — one renderer,
one camera, one panel, so the four are directly comparable:

```bash
python -m Centralized_Local_Planner.main replan --method stop_and_go
python -m Centralized_Local_Planner.main replan --method speed_adjusting
python -m Centralized_Local_Planner.main replan --method optimization_based
python -m Centralized_Local_Planner.main replan --method learning_based
```

Add `--preview` to open a live window instead of writing an `.mp4` (Steps A–D
only). Rendered files go to `../outputs/` — see `../outputs/README.md`.

## Programmatic pipeline

`Pipeline.step(frame)` runs A→B→C→D for one frame and returns plain data
structures instead of drawing — the integration point for the local layer:

```python
from Centralized_Local_Planner.main import Pipeline

pipe = Pipeline(num_frames=360, num_workers=2, num_amrs=6)
for f in range(pipe.num_frames):
    out = pipe.step(f)
    # out["worker_data"] : per-worker safety tubes          (Step B)
    # out["results"]     : {amr_name: ConflictResult}       (Step C status / TTC)
    # out["clusters"]    : ClusterResult                    (Step D groups + region)
```

`Pipeline(replanner=...)` additionally runs a rail-style replanner in the loop —
that is how `local_path_replanning/speed_adjusting/` is driven. The other three
methods use their own closed loop,
`local_path_replanning.common.runtime.LocalReplanningRuntime`, which adds the
cluster hand-over and the command tracking the shared action space needs.

## The local layer

Four methods, sharing one runtime, one rollout and one space-time reservation
shield, differing only in what they may do and who decides:

| method | action set | who chooses | DOF |
|---|---|---|---|
| `stop_and_go` | drive at full speed, or halt | fixed rule | 0 |
| `speed_adjusting` | speed × {1, ⅔, ⅓, 0}, on the rail | fastest admissible | 1 |
| `optimization_based` | `(goal_fwd, goal_lat, speed_scale)` | lowest-cost safe candidate | 3 |
| `learning_based` | `(goal_fwd, goal_lat, speed_scale)` | attention policy | 3 |

Because the shield is the same code behind all four, a comparison between them
measures coordination and throughput, never safety.

`optimization_based` currently holds a greedy stand-in and is the slot for an
MPC formulation. See `local_path_replanning/README.md` for the action spaces,
the interfaces, the deviations from the specification, and how to run
everything.

## History

Two earlier lines of work were removed once this structure landed; their code is
in git history and their rendered results are frozen under `../archive/`:

* the **rail V1** attention policy that proposed a speed factor behind the
  speed-adjusting shield (`rl/fleet_env.py`, `rl/policy.py`, `rl/train_v1.py`);
* the **spatial** replanner that let cluster members leave their rails and move
  freely in 2D inside a locked busy area (`tools/local_replanning.py`,
  `rl/episodic_env.py`, `viz/render_local.py`).

Both predate the shared action space and shield, so their numbers were never
comparable with the four methods above. See `../archive/README.md`.
