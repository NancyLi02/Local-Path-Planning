# Local path replanning

The local layer of the centralized planner. Steps A–D (`../tools`) predict where
the workers will be, inflate that into no-go lobes, mark which AMRs are
affected, and group them into conflict clusters. This package is what happens
next: given a cluster of `N <= max_agents` AMRs it decides, every frame, what
each of them should actually do.

```
policy proposes an efficient action   ->   safety shield decides what may execute
```

## Four methods, one ladder

They share the runtime, the rollout, the space-time reservation shield and the
command format, and differ only in **what they may do** and **who decides**.
That is the point: because the shield is the same code behind all four, a
comparison between them measures coordination and throughput, never safety.

| | method | action set | who chooses | DOF |
|---|---|---|---|---|
| a | `stop_and_go/` | drive at full speed, or halt | fixed rule | 0 |
| b | `speed_adjusting/` | speed × {1, ⅔, ⅓, 0}, on the rail | fastest admissible | 1 |
| c | `optimization_based/` | `(goal_fwd, goal_lat, speed_scale)` | lowest-cost safe candidate | 3 |
| d | `learning_based/` | `(goal_fwd, goal_lat, speed_scale)` | attention policy | 3 |

**c and d have the identical action space**, so the difference between their
numbers isolates what the *learned proposal* is worth, with nothing else
varying. c is also what d is behaviour-cloned from.

> **c is a placeholder.** `optimization_based/planner.py` currently holds a
> greedy stand-in: a nominal "drive to the local goal at full speed" proposal
> expanded into a candidate set and scored through the shield. It is the slot
> for a proper MPC formulation. Replacing it touches nothing else — not the
> runtime, the shield, the harness, the renderer, or `main.py`.

## Layout

```
local_path_replanning/
├── registry.py              the four methods; everything that switches resolves here
├── common/                  shared by all four
│   ├── cluster_state.py     the data contract with the simulator (or a real fleet)
│   ├── config.py            every tunable constant
│   ├── observation_builder.py  per-AMR feature vector for the learned policy
│   ├── action_decoder.py    raw -> (goal_fwd, goal_lat, speed_scale)
│   ├── trajectory_rollout.py   action -> trajectory + backup candidates
│   ├── safety_shield.py     the four checks, the candidate cost, the choice
│   ├── commands.py          what the planner hands the tracking controller
│   ├── planner_base.py      rollout -> backups -> shield -> dispatch
│   ├── worker_cache.py      pre-computed Step-A/B worker tubes
│   └── runtime.py           the closed loop: A-D + replanning + execution
├── stop_and_go/planner.py
├── speed_adjusting/
│   ├── planner.py           the replanner itself
│   ├── adapter.py           runs it in the shared harness (metrics + renderer)
│   └── evaluate.py          its own no-shield baseline + shield-parameter tuning
├── optimization_based/planner.py
├── learning_based/
│   ├── policy.py            encoder -> self-attention -> actor + centralized critic
│   ├── planner.py
│   ├── env.py               one episode = one conflict cluster
│   └── train.py             BC warm-start + curriculum PPO
├── evaluate.py              cross-method comparison harness
├── sweep.py                 configuration ablations
├── render.py                one renderer, any method
├── plot_curve.py
└── report/{build,diagrams}.py
```

### Adding a fifth method

Subclass `common.planner_base.ShieldedReplanner`, override `propose_actions`
(and `candidate_actions` if the action set changes), add it to `registry.py`.
The shield, rollout, commands, evaluation, ablation and rendering come free.

## Interfaces

```python
from Centralized_Local_Planner.local_path_replanning import make_planner
planner  = make_planner("learning_based", cfg, model="logs/.../best.pt")
commands = planner.plan(cluster, worker_predictions, map_data, dt)
# commands[amr_id] -> Command(mode="TRACK"|"STOP", waypoints, target_speed, speed_limit, ...)
```

* `cluster` — list of `ClusterAgent` (id, reference path, `s`, lateral offset,
  speed, TTC, task priority, braking risk, affected flag).
* `worker_predictions` — list of `WorkerPrediction` (Step-A ellipses + Step-B
  hard/soft lobes, time-aligned by look-ahead time).
* `map_data` — static workstation rectangles + map bounds.

Observation blocks (per AMR): ego 6, local goal 2, route preview 3×10, worker
tube 7×3, priority 5, plus an optional 4-dim spatial token — 68 total.
Action: `(goal_fwd, goal_lat, speed_scale)`, decoded `sigmoid / tanh / sigmoid`.

## Running

```bash
M=Centralized_Local_Planner.local_path_replanning

# the whole ladder in one table
python -m $M.evaluate --methods stop_and_go,speed_adjusting,optimization_based,learning_based \
       --seeds 5 --frames 560 \
       --out outputs/5_local_path_replanning/results/comparison.json

# one method
python -m $M.evaluate --method optimization_based --seeds 5
python -m $M.evaluate --method learning_based --model logs/local_path_replanning/learning_based/best.pt

# demos -- one renderer, one camera, so they are comparable
python -m $M.render --method stop_and_go
python -m $M.render --method speed_adjusting
python -m $M.render --method optimization_based
python -m $M.render --method learning_based
#   equivalently, through the framework CLI:
python -m Centralized_Local_Planner.main replan --method learning_based

# speed_adjusting's own study: is the shield worth anything, and at what horizon?
python -m $M.speed_adjusting.evaluate --seeds 5

# design decisions, one factor at a time
python -m $M.sweep --ablation --seeds 5 --jobs 5

# training + curve + write-up
python -m $M.learning_based.train --timesteps 25000 --lr 5e-5 --bc-coef 0.5
python -m $M.plot_curve
python -m $M.report.build
```

`--set key=value` on `evaluate` overrides any `PlannerConfig` field; that is how
the ablations are run.

Results land in `outputs/5_local_path_replanning/`, checkpoints in
`logs/local_path_replanning/`. Worker tubes are cached under `.cache/` and
regenerate themselves when missing.

## Where this deviates from the specification, and why

Every deviation is switchable from `PlannerConfig`. The ablation table
(`sweep.py --ablation`) is what settles each one; three earlier deviations were
**withdrawn** once a bug in the manoeuvre kinematics was fixed, which is why the
ablation is part of the deliverable rather than a footnote.

1. **Observation is 68-dim, not 64.** The specified feature set is entirely
   path-relative, so two AMR tokens carry nothing locating them relative to each
   other and the self-attention cannot express AMR–AMR coordination — the one
   thing it is there for. A 4-dim spatial token (normalised x, y, cos/sin
   heading) is appended. `include_spatial_block=False` restores the exact 64-dim
   specification vector.
2. **Holonomic lateral control, driven in time.** The lateral offset follows
   `l' = clip(k (goal_lat - l), ±v_lat_max)`, sharing the speed budget with the
   longitudinal rate, rather than an arc-length ramp. An arc-length ramp cannot
   move a *halted* AMR sideways at all, removing its only escape at exactly the
   moment it needs one; and a ramp whose slope starts at zero never accumulates
   any lateral motion under 5 Hz replanning. The AMRs are holonomic in this
   factory model.
3. **Kinematic limits are applied to the manoeuvre, on the velocity vector.**
   The global reference path has 90° corners the fleet already drives; charging
   them to the local planner rejects every trajectory. And limiting the two axes
   independently allows a combined `sqrt(2)·a_max`, which the feasibility check
   then rejects — silently making every forward candidate infeasible whenever a
   lateral correction was active. This was the bug behind the freezing
   behaviour that motivated the withdrawn deviations below.
4. **Emergency reverse** (`allow_emergency_reverse`). The policy action space is
   forward-only, per the specification. A boxed-in AMR has no escape left, so
   the *shield* (never the policy) may offer a slow reverse, and only when no
   candidate is safe.
5. **Least-unsafe fallback** (`least_unsafe_fallback`). When nothing is safe the
   specification says STOP. Standing still is what gets an AMR run over, so
   candidates are ranked by violation severity and clearance instead; the STOP
   candidate is ranked alongside the others and still wins whenever moving would
   be worse.
6. **Two extra "short detour" candidates** (`use_detour_candidates`), realising
   the command vocabulary's detour command: a 0.5 m shift often cannot clear a
   worker beside the lane.
7. **Dropout is 0.** PPO samples actions with the network in eval mode and
   re-evaluates them in train mode, so any dropout makes the importance ratio
   compare two different networks. Measured: return decays 36 → 14 over 10k
   steps (`archive/logs/step_e_v1/ppo_dropout_run.log`).
8. **Lazy backup expansion** (`lazy_backups`). A planner that executes its own
   proposal when it is safe only rolls out the backup set when the proposal
   fails. Identical commands, fewer rollouts — this is where a learned proposal
   pays off in compute.
9. **The rule methods accept the lowest-cost safe candidate; the learned one
   executes its own proposal whenever it is safe.** That is what the
   specification asks for in each case, and it makes `shield_modified` mean "the
   policy was genuinely overridden", which is what the `r_shield` reward term
   needs.

### Withdrawn (the specification was right)

Introduced to fight the freezing behaviour, kept until the ablation was run,
then reverted when the real cause turned out to be the acceleration bug in
item 3:

* *Hard constraint over a 1.5 s window instead of the full 5 s tube.* Measured
  strictly worse. `hard_horizon_sec` is back to the full prediction horizon.
* *Worker keep-out reduced from `r_amr + r_worker` = 1.25 m to 0.85 m.* No
  measurable difference; reverted to the specified 1.25 m.
* *Command-consistency (switching) cost.* Cost efficiency for no safety gain
  once the kinematics were right. `w_switch = 0` by default, kept as an ablation.

## Results

**The measured numbers are not in this file.** They were produced by the
pre-reorganisation package and are frozen under
`archive/outputs/8_step_e_v1_module/` together with the write-up that reasons
about them; the method names there are the old `V0` / `V1` / `rail V0`. Run the
comparison above to regenerate them under the current names, then
`report.build` to rebuild the page.

What the archived run established, and what should be re-checked after the
teacher is replaced with MPC:

* Safety is the shield's job. Every collision-free rung scores exactly 0.00,
  from drive-or-halt upwards; cutting the action set down does not make the
  fleet less safe.
* Speed control alone is already collision-free and buys most of the throughput
  back from stop-and-go.
* The lateral degree of freedom buys the rest — about 12 % of mission time and
  three quarters of the stopping.
* The learned proposal more than halved the remaining stopping again, at equal
  safety and completion, but did **not** shorten the mission: the win was
  smoother motion, not a faster fleet. With the teacher already at zero
  collisions and full completion, the shield-constrained optimum was close to
  attained and PPO had little left to gain. That ceiling is the reason to
  replace the greedy teacher with MPC.
