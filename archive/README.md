# archive/ — frozen results from earlier attempts at the local layer

Nothing in here is regenerable: the code that produced most of it was removed
when the local replanning layer was reorganised into
`Centralized_Local_Planner/local_path_replanning/` (four methods behind one
shared safety shield). It is kept because the June write-up and the slide deck
cite it, and because it is the record of what was tried.

**Do not compare these numbers with `outputs/5_local_path_replanning/.`** Each
generation used a different action space, a different observation and a
different shield.

## What is here

| folder | what it was | code |
|---|---|---|
| `outputs/5_step_e_rail_v0_speed/` | rail V0: speed-only shield, AMRs never leave the rail | **still exists**, as `local_path_replanning/speed_adjusting/` |
| `outputs/6_step_e_rail_v1_rl/` | rail V1: attention RL proposing a speed factor behind that shield | removed (`Centralized_Local_Planner/rl/{fleet_env,policy,train_v1}.py`) |
| `outputs/7_step_e_spatial/` | spatial: cluster AMRs leave the rail and move in 2D, rule and RL versions | removed (`tools/local_replanning.py`, `rl/episodic_env.py`, `viz/render_local.py`) |
| `outputs/8_step_e_v1_module/` | the specification's Step E: 3-D action, rollout, four-check shield | **still exists**, reorganised into the four methods |
| `outputs/slides/` | earlier presentation deck | — |
| `logs/V1/` | rail V1 checkpoints + TensorBoard | removed |
| `logs/V1_local/` | spatial RL checkpoints + TensorBoard | removed |
| `logs/step_e_v1/` | training run logs + TensorBoard for `8_` | the checkpoints stayed live, in `logs/local_path_replanning/learning_based/` |

Two of these logs are cited as evidence in the current
`local_path_replanning/README.md` and are the reason `logs/step_e_v1/` is kept
rather than deleted:

* `logs/step_e_v1/ppo_dropout_run.log` — dropout during PPO makes the importance
  ratio compare two different networks; return decays 36 → 14 over 10k steps.
* `logs/step_e_v1/vanilla_ppo_run.log` — PPO without the behaviour anchor.

## Naming

`outputs/8_step_e_v1_module/` uses the pre-reorganisation method names. The
mapping to the current ones:

| there | here now |
|---|---|
| `stopgo` / STOP-GO | `stop_and_go` |
| `rail_v0` / rail V0 | `speed_adjusting` |
| `v0` / V0 | `optimization_based` |
| `v1` / V1 | `learning_based` |

Note the collision that made the old numbering confusing: the *rail* V1 in
folder `6_` and the module's V1 in folder `8_` were different methods that were
both called "V1". The folder is what tells them apart.

To recover anything removed:

```bash
git log --diff-filter=D --name-only -- 'Centralized_Local_Planner/rl/*'
git show <commit>^:Centralized_Local_Planner/rl/fleet_env.py
```
