# outputs/ — what is in here

Every rendered demo and every metrics file, filed by the pipeline step that
produced it. Folders are numbered in pipeline order, so `1_` … `4_` are the
perception and clustering stages and `5_` … `8_` are four *different* attempts
at Step E, in the order they were built.

| folder | step | produced by | what it shows |
|---|---|---|---|
| `1_step_a_prediction/` | A | `Centralized_Local_Planner/viz/render_prediction.py` | worker intent prediction, 2 and 4 workers |
| `2_step_b_safety_inflation/` | B | `viz/render_safety.py` | hard / soft no-go lobes + still frames |
| `3_step_c_affected_amr/` | C | `viz/render_affected.py` | per-AMR conflict status, TTC, Gantt stills |
| `4_step_d_conflict_cluster/` | D | `viz/render_clusters.py` | conflict clusters and their busy-area hulls |
| `5_step_e_rail_v0_speed/` | E | `tools/replanning.py`, `eval_v0.py` | **rail V0**: speed-only shield, AMRs never leave the rail |
| `6_step_e_rail_v1_rl/` | E | `rl/{fleet_env,policy,train_v1}.py` | **rail V1**: attention RL proposing a speed factor behind that shield |
| `7_step_e_spatial/` | E | `tools/local_replanning.py`, `rl/episodic_env.py` | **spatial**: cluster AMRs leave the rail and move in 2D (rule + RL) |
| `8_step_e_v1_module/` | E | `step_e_v1/` | **the specification's Step E** (CLAUDE.md): 3-D action, trajectory rollout, four-check shield, command dispatch |
| `slides/` | — | — | earlier presentation deck |

## Which folder should I look at?

* **The current deliverable is `8_step_e_v1_module/`.** Start with `report.html`,
  then `results/compare_3way.json`. Everything in it comes from the `step_e_v1/`
  package and is reproducible with the commands in `step_e_v1/README.md`.
* `5_`, `6_` and `7_` are the earlier Step-E attempts, kept because they are what
  the June write-up and the slide deck refer to. They are **not** produced by
  `step_e_v1/` and their numbers are not comparable with it: they use a
  different action space, a different observation and a different shield.
* Two names collide across generations by history: the *rail* V1 in `6_` and the
  module's V1 in `8_` are different methods that were both called "V1". The
  folder tells them apart; the file names inside no longer repeat the prefix.

## `8_step_e_v1_module/` in detail

```
report.html                       the write-up (published artifact)
training_curve.png                BC + PPO against the V0 teacher
demos/demo_stopgo.mp4             stop-and-go baseline, closed loop
demos/demo_v0.mp4                 V0, closed loop
demos/demo_v1.mp4                 V1, closed loop
results/compare_3way.json         MAIN TABLE - stop-and-go / V0 / V1, 5 seeds x 560 frames
results/ablation.json             one-factor ablation, 10 configurations
results/stopgo.json               stop-and-go alone
results/v0_speed_only.json        V0 with the lateral degree of freedom removed
results/v0.json                   V0 (speed + lateral), the baseline
results/v1_safety_first.json      V1, proposal ranked among the candidates
results/v1_proposal_first.json    V1, proposal executed whenever it is safe
results/compare_v0_v1_420f.json   earlier V0/V1 run at the 420-frame horizon
results/sweep_shield_params.json  exploratory sweep: shield horizons, lateral speed
results/sweep_candidate_set.json  exploratory sweep: candidate set, AMR clearance
```

Model checkpoints and training logs for the module live in `logs/step_e_v1/`,
not here.

## Regenerating

Each folder's contents are written by its own command; defaults now point at
the folder, so nothing lands loose in `outputs/` again.

```bash
python -m Centralized_Local_Planner.main pipeline          # 1_ .. 4_
python -m Centralized_Local_Planner.eval_v0                # 5_
python -m Centralized_Local_Planner.rl.eval_v1             # 6_
python -m Centralized_Local_Planner.rl.eval_local_compare  # 7_
python -m step_e_v1.evaluate --planners stopgo,v0,v1 \
       --model logs/step_e_v1/v1_best.pt --seeds 5 --frames 560 \
       --out outputs/8_step_e_v1_module/results/compare_3way.json
python -m step_e_v1.render --planner v1 --model logs/step_e_v1/v1_best.pt
```
