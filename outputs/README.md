# outputs/ — what is in here

Every rendered demo and every metrics file produced by the **current** code,
filed by the pipeline step that produced it.

| folder | step | produced by | what it shows |
|---|---|---|---|
| `1_step_a_prediction/` | A | `Centralized_Local_Planner/viz/render_prediction.py` | worker intent prediction, 2 and 4 workers |
| `2_step_b_safety_inflation/` | B | `viz/render_safety.py` | hard / soft no-go lobes + still frames |
| `3_step_c_affected_amr/` | C | `viz/render_affected.py` | per-AMR conflict status, TTC, Gantt stills |
| `4_step_d_conflict_cluster/` | D | `viz/render_clusters.py` | conflict clusters and their busy-area hulls |
| `5_local_path_replanning/` | local | `Centralized_Local_Planner/local_path_replanning/` | the four local replanning methods, compared |

Results from the **earlier** attempts at the local layer are frozen under
`../archive/outputs/` — see `../archive/README.md`. They are not comparable with
`5_` and the code that produced most of them no longer exists.

## `5_local_path_replanning/` in detail

```
report.html                      the write-up
training_curve.png               BC + PPO against the teacher
demos/demo_stop_and_go.mp4       a. drive or halt
demos/demo_speed_adjusting.mp4   b. speed steps on the rail
demos/demo_optimization_based.mp4  c. speed + lateral, cost search   (MPC slot)
demos/demo_learning_based.mp4    d. speed + lateral, attention policy
demos/web/                       900 px versions embedded in report.html
results/comparison.json          MAIN TABLE - all four methods, one run
results/ablation.json            one-factor ablation
results/speed_adjusting.json     method b measured on its own
results/speed_adjusting_tuning.json   its no-shield baseline + shield-horizon search
results/sweep.json               exploratory configuration sweep
```

All four demo videos are rendered by the same renderer on the same seed, so
they are directly comparable — the planner is the only thing that changes.

Model checkpoints and training logs live in `../logs/local_path_replanning/`,
not here.

## Regenerating

```bash
M=Centralized_Local_Planner.local_path_replanning

# 1_ .. 4_
python -m Centralized_Local_Planner.main pipeline

# 5_ : the comparison table
python -m $M.evaluate \
       --methods stop_and_go,speed_adjusting,optimization_based,learning_based \
       --seeds 5 --frames 560 \
       --out outputs/5_local_path_replanning/results/comparison.json

# 5_ : the four demos
for m in stop_and_go speed_adjusting optimization_based learning_based; do
    python -m $M.render --method $m
done

# 5_ : ablation, curve, write-up
python -m $M.sweep --ablation --seeds 5 --jobs 5 \
       --out outputs/5_local_path_replanning/results/ablation.json
python -m $M.plot_curve
python -m $M.report.build
```

Every command writes into its own folder by default, so nothing lands loose in
`outputs/` again.
