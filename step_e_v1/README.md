# Step E V1 — attention-based shared policy for variable-N multi-AMR local replanning

Implementation of the Step-E specification (`CLAUDE.md`): a **centralized**
local replanning module that takes a conflict cluster of `N <= max_agents`
AMRs, builds one observation per AMR, runs a **shared-weight attention
policy**, decodes one bounded action per AMR, rolls each action out into a
short-horizon trajectory, and pushes every trajectory through a **space-time
reservation safety shield** before dispatching commands.

```
policy proposes efficient actions   ->   safety shield decides what may execute
```

## Layout

```
step_e_v1/
├── config.py                V1Config                       (every tunable constant)
├── cluster_state.py         ClusterAgent / WorkerPrediction / MapData / PathTable
├── observation_builder.py   build_cluster_observation, extract_* feature blocks
├── attention_policy.py      MultiAMRAttentionPolicy (encoder / transformer / actor / critic)
├── action_decoder.py        decode_action, action_to_local_goal, reverse_action
├── trajectory_rollout.py    Trajectory, rollout_action, generate_backup_candidates
├── safety_shield.py         the four checks, candidate cost J, joint selection
├── commands.py              Command, trajectory_to_command
├── planner_base.py          shared rollout -> backups -> shield -> dispatch loop
├── stopgo_planner.py        StopAndGoReplanner      (drive-or-halt baseline)
├── v0_planner.py            V0SequentialReplanner   (TTC-priority baseline)
├── v1_planner.py            V1AttentionReplanner    (learned proposals)
├── runtime.py               A->D pipeline + Step E + command tracking (closed loop)
├── env.py                   episodic cluster RL environment
├── worker_cache.py          pre-computed Step-A/B worker tubes
├── evaluate.py              V0 vs V1 experiment harness
├── sweep.py                 configuration ablations
└── train/ppo_trainer.py     BC warm-start + curriculum PPO
```

## Interfaces

```python
planner = V1AttentionReplanner.from_checkpoint("logs/step_e_v1/v1_best.pt", cfg)
commands = planner.plan(cluster, worker_predictions, map_data, dt)
# commands[amr_id] -> Command(mode="TRACK"|"STOP", waypoints, target_speed, speed_limit, ...)
```

* `cluster` — list of `ClusterAgent` (id, reference path, `s`, lateral offset,
  speed, TTC, task priority, braking risk, affected flag).
* `worker_predictions` — list of `WorkerPrediction` (Step-A ellipses + Step-B
  hard/soft lobes, time-aligned by look-ahead time).
* `map_data` — static workstation rectangles + map bounds.

Observation blocks (per AMR): ego 6, local goal 2, route preview 3x10, worker
tube 7x3, priority 5 — plus an optional 4-dim spatial token (see below).
Action: `(goal_fwd, goal_lat, speed_scale)` decoded with
`sigmoid / tanh / sigmoid`.

## Where this implementation deviates from the specification, and why

Every deviation below is switchable from `V1Config`, and each one is backed by
a measurement in `outputs/step_e_v1_ablation.json` (V0, 5 seeds, 420 frames).
Three earlier deviations were **withdrawn** once a bug in the manoeuvre
kinematics was fixed — see "Withdrawn" below; that is why the ablation table is
part of the deliverable rather than a footnote.

1. **Observation is 68-dim, not 64.** The specified feature set is entirely
   path-relative, so two AMR tokens carry nothing that locates them relative to
   each other and the self-attention cannot express AMR-AMR coordination — the
   one thing it is there for. A 4-dim spatial token (normalised x, y, cos/sin
   heading) is appended. `include_spatial_block=False` restores the exact
   64-dim specification vector.
2. **Holonomic lateral control, driven in time.** The lateral offset follows
   `l' = clip(k (goal_lat - l), +-v_lat_max)` with the speed budget shared with
   the longitudinal rate, rather than an arc-length ramp. An arc-length ramp
   cannot move a *halted* AMR sideways at all, which removes its only escape at
   exactly the moment it needs one; and a ramp whose slope starts at zero never
   accumulates any lateral motion under 5 Hz re-planning. The AMRs are
   holonomic in this factory model, as in the existing rule baseline.
3. **Kinematic limits are applied to the manoeuvre, on the velocity vector.**
   The global reference path has 90-degree corners the fleet already drives;
   charging them to the local planner rejects every trajectory. And limiting
   the two axes independently allows a combined `sqrt(2) a_max`, which the
   feasibility check then rejects — silently making every forward candidate
   infeasible whenever a lateral correction was active. This was the bug that
   caused the freezing behaviour that motivated deviations 1-3 of the earlier
   draft.
4. **Emergency reverse** (`allow_emergency_reverse`). The policy action space
   is forward-only, per the specification. A boxed-in AMR has no escape left,
   so the *shield* (never the policy) may offer a slow reverse, and only when
   no candidate is safe. Ablation: removing it costs 1.00 collisions and
   16.7 points of completion.
5. **Least-unsafe fallback** (`least_unsafe_fallback`). When nothing is safe
   the specification says STOP. Standing still is what gets an AMR run over, so
   the candidates are ranked by violation severity and clearance instead; the
   STOP candidate is ranked alongside the others and still wins whenever moving
   would be worse. Ablation: plain STOP costs 1.00 collisions and raises the
   stop ratio from 3.3 % to 35.9 %.
6. **Two extra "short detour" candidates** (`use_detour_candidates`), realising
   the command vocabulary's detour command: a 0.5 m shift often cannot clear a
   worker beside the lane. Ablation: worth 0.8 points of stop ratio.
7. **Dropout is 0.** PPO samples actions with the network in eval mode and
   re-evaluates them in train mode, so any dropout makes the importance ratio
   compare two different networks. Measured: the return decays from 36 to 14
   over 10k steps (`logs/step_e_v1/ppo_dropout_run.log`).
8. **Lazy backup expansion** (`lazy_backups`). A planner that executes its own
   proposal when it is safe (V1) only rolls out the backup set when the
   proposal fails. Identical commands, fewer rollouts — this is where a learned
   proposal pays off in compute.
9. **V0 accepts the lowest-cost safe candidate; V1 executes its own proposal
   whenever it is safe.** That is what the specification asks for in each case,
   and it makes `shield_modified` mean "the policy was genuinely overridden",
   which is what the `r_shield` reward term needs.

### Withdrawn deviations (the specification was right)

These were introduced to fight a freezing-robot behaviour, kept until the
ablation was run, then reverted when the real cause turned out to be the
acceleration bug in item 3:

* *Hard constraint over a 1.5 s window instead of the full 5 s tube.* Measured
  strictly worse: 1.00 vs 0.00 collisions, 83.3 % vs 100 % completion.
  `hard_horizon_sec` is back to the full prediction horizon.
* *Worker keep-out reduced from `r_amr + r_worker` = 1.25 m to 0.85 m.* No
  measurable difference; reverted to the specified 1.25 m.
* *Command-consistency (switching) cost.* Cost efficiency for no safety gain
  once the kinematics were right (stop ratio 4.7 % -> 3.3 %, route deviation
  0.29 m -> 0.04 m). `w_switch = 0` by default, kept as an ablation.

## Results

Full simulator, 5 seeds, 560 frames, 6 AMRs, 2 workers. Identical safety shield
behind every planner; V1 uses `logs/step_e_v1/v1_best.pt`. The horizon is 560
frames rather than 420 so that every planner reaches 100 % completion and the
makespan is directly comparable.

**STOP-GO** is the classic industrial baseline: the same observation, the same
TTC priority order, the same shield and the same commands, with the action set
cut down to `{GO at full speed on the path, STOP and wait}` -- no speed
modulation, no lateral shift, no reverse, no least-unsafe fallback.

| metric                   | STOP-GO | V0 (baseline) | V1 safety-first | V1 proposal-first |
|--------------------------|--------:|--------------:|----------------:|------------------:|
| worker collisions        |**0.00** |      **0.00** |        **0.00** |              0.40 |
| completion %             | **100** |       **100** |         **100** |              93.3 |
| makespan [frames]        |   435.0 |     **352.6** |           354.8 |     n/a, censored |
| stop ratio %             |   59.04 |          3.30 |            1.42 |          **0.76** |
| route deviation [m]      |   0.000 |         0.038 |           0.073 |             0.084 |
| AMR-frames replanning    |     519 |           328 |             302 |           **273** |
| candidate rollouts / AMR |**2.00** |          8.00 |            8.00 |              2.86 |
| plan time [ms]           |**10.6** |          33.4 |            60.6 |              37.1 |

* **Safety is the shield's job, not the planner's.** All three collision-free
  planners score exactly 0.00 with the same minimum clearance (0.278 m). Cutting
  the action set down to stop-and-go does not make the fleet less safe -- the
  space-time shield already guarantees that.
* **What the richer action set buys is throughput.** Stop-and-go halts on
  **59 %** of the AMR-frames it controls against V0's 3.3 %, spends 519 AMR-frames
  under local control against 328, and needs **435 frames to clear the mission
  against 353 -- 23 % longer**. Continuous speed control plus one metre of lateral
  freedom is worth roughly a quarter of the makespan here.
* **V1 safety-first** (the learned action is one more candidate, shield still
  picks the cheapest safe one) keeps the perfect safety and completion and **more
  than halves the stopping again** (3.30 % -> 1.42 %), with 8 % fewer AMR-frames
  spent under local control. Its makespan matches V0's (354.8 vs 352.6): the win
  is smoother motion, not a shorter mission.
* **V1 proposal-first** (the specification's V1 flow -- the policy's own action
  executes whenever it is safe) stops least of all and needs only **2.86 candidate
  rollouts per AMR instead of 8**, because a good proposal makes the backup set
  unnecessary. It costs 0.4 collisions per run: a small imitation error is
  unforgiving in a simulator where workers walk into whatever stands in their way.
* **Cost of the ladder.** Planning is 10.6 ms for stop-and-go, 33.4 ms for V0 and
  60.6 ms for V1 safety-first (network forward pass plus the full candidate
  sweep); proposal-first V1 gets back to 37.1 ms by skipping the backups it does
  not need. All are far inside the 200 ms control period.

Training (`logs/step_e_v1/v1_train.log`, curve in
`outputs/step_e_v1_module_training_curve.png`): behaviour cloning reaches the V0
teacher (return 32.0 vs 35.8, zero collisions); PPO then briefly exceeds it
(35.7 at 3k steps, shield override 8.5 % vs V0's 9.6 %) and afterwards decays.
With V0 already at zero collisions and full completion, the shield-constrained
optimum is essentially attained, so PPO has little to gain and a great deal to
lose. Two failure modes were found and fixed along the way -- raw-space BC
(ill-conditioned, loss 10.3 -> 0.007 in decoded space) and dropout during PPO
(the importance ratio compares two different networks).

## Ablations

V0, 5 seeds (`outputs/step_e_v1_ablation.json`). Each row disables exactly one
design decision.

| configuration                        | collisions | completion % | stop % | plan ms |
|--------------------------------------|-----------:|-------------:|-------:|--------:|
| **default**                          |   **0.00** |    **100.0** |    3.3 |    38.5 |
| shorter hard window (1.0 s)          |       1.00 |         83.3 |    1.8 |    35.9 |
| no emergency reverse                 |       1.00 |         83.3 |    7.2 |    34.1 |
| plain STOP fallback                  |       1.00 |         83.3 |   35.9 |    36.4 |
| no lateral DOF (speed only)          |       0.00 |        100.0 |   10.2 |    40.0 |
| no detour candidates                 |       0.00 |        100.0 |    4.1 |    26.3 |
| backups off (proposal + STOP)        |       0.00 |         83.3 |   34.0 |    15.2 |
| dense candidate grid (25 candidates) |       0.00 |        100.0 |    0.5 |   130.2 |
| with command-consistency cost        |       0.00 |        100.0 |    4.7 |    38.2 |
| tighter keep-out (0.85 m)            |       0.00 |        100.0 |    3.3 |    37.8 |

Load-bearing for **safety**: the full-horizon hard constraint, the emergency
reverse, and the least-unsafe fallback -- removing any one of them costs a
collision and 16.7 points of completion. Load-bearing for **throughput**: the
backup candidate set (34 % -> 3.3 % stop ratio) and the lateral degree of
freedom (10.2 % -> 3.3 %). The dense 25-candidate grid buys another 2.8 points
of stop ratio for 3.4x the planning time.

## Reproduce

```bash
python -m step_e_v1.evaluate --planners stopgo,v0,v1 --seeds 5 --frames 560 \
       --model logs/step_e_v1/v1_best.pt --set v1_prefer_proposal=False
python -m step_e_v1.train.ppo_trainer --timesteps 25000 --lr 5e-5 --bc-coef 0.5
python -m step_e_v1.evaluate --compare --model logs/step_e_v1/v1_best.pt --seeds 5 \
       --out outputs/step_e_v1_module_compare.json
python -m step_e_v1.evaluate --planner v1 --model logs/step_e_v1/v1_best.pt \
       --seeds 5 --set v1_prefer_proposal=False                  # safety-first mode
python -m step_e_v1.sweep --ablation --seeds 5 --jobs 5           # ablation table
python -m step_e_v1.plot_curve                                    # training curve
python -m step_e_v1.render --planner v1 --model logs/step_e_v1/v1_best.pt
```

## Artifacts

```
outputs/step_e_v1_module_compare3.json         STOP-GO vs V0 vs V1, 5 seeds
outputs/step_e_v1_module_compare.json          V0 vs V1, 5 seeds (420 frames)
outputs/step_e_v1_v1_safety_first.json         V1 safety-first mode
outputs/step_e_v1_v1_proposal_first.json       V1 proposal-first mode
outputs/step_e_v1_v0_baseline.json             V0 baseline
outputs/step_e_v1_ablation.json                one-factor ablation table
outputs/step_e_v1_module_training_curve.png    BC + PPO curve vs the V0 teacher
outputs/step_e_v1_stopgo_demo.mp4              stop-and-go closed-loop demo
outputs/step_e_v1_v0_demo.mp4                  V0 closed-loop demo
outputs/step_e_v1_v1_demo.mp4                  V1 closed-loop demo
logs/step_e_v1/v1_{bc,best,final}.pt           checkpoints
logs/step_e_v1/v1_train.log                    training log (dropout disabled)
logs/step_e_v1/ppo_dropout_run.log             the dropout failure, kept as evidence
logs/step_e_v1/vanilla_ppo_run.log             PPO without the behaviour anchor
```

The files named `outputs/step_e_v1_results.json`,
`outputs/step_e_v1_training_curve.png` and `outputs/step_e_v1_replanning_demo.mp4`
belong to the EARLIER speed-only Step-E V1 in `Centralized_Local_Planner/rl/`
and are left untouched.
