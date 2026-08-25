# CLAUDE.md

## Project Context

This project implements **Step E: variable-N multi-AMR local path replanning** for a factory navigation system.

The planner receives a conflict cluster containing a variable number of autonomous mobile robots (AMRs), usually:

\[
N \in \{1,2,3,4\}
\]

The goal is to generate short-horizon local replanning commands for all AMRs in the conflict cluster while avoiding predicted worker occupied regions, static obstacles, and AMR-AMR collisions.

The system should support two methods:

1. **V0: TTC-priority sequential replanning baseline**
2. **V1: attention-based shared policy for variable-N multi-AMR replanning**

V0 is the reliable engineering baseline. V1 is the learning-based method designed to improve joint coordination quality.

The key design principle is:

\[
\text{policy proposes efficient actions, safety shield decides what can be executed}
\]

---

# Step E Goal

Step E solves centralized multi-AMR local replanning under dynamic worker safety constraints.

Given a conflict cluster:

\[
\mathcal{C} = \{1,2,\ldots,N\}
\]

where each AMR has its current state, global path reference, local goal, velocity, TTC, and task priority, Step E outputs one local command per AMR:

\[
\{u_i\}_{i=1}^{N}
\]

Each command may contain:

- local waypoints;
- target speed;
- speed limit;
- stop command;
- lane shift command;
- short detour command.

The planner must satisfy:

1. AMR-worker safety;
2. AMR-AMR safety;
3. static obstacle safety;
4. dynamic feasibility;
5. low route deviation;
6. low delay;
7. real-time computation.

This is not standard decentralized MARL. The system uses centralized planning and centralized execution. Therefore, the method should be implemented as a centralized variable-N local replanning module.

---

# V0: TTC-Priority Sequential Replanning Baseline

## V0 Overview

V0 is the first working method and also the main baseline.

The idea is:

1. Sort AMRs by urgency.
2. Plan AMRs one by one.
3. After one AMR trajectory is accepted, reserve its future space-time region.
4. Later AMRs treat earlier accepted trajectories as dynamic obstacles.
5. Every planned trajectory must pass the safety shield.

The priority score can be defined as:

\[
\text{priority}_i =
\alpha \cdot \frac{1}{\text{TTC}_i + \epsilon}
+ \beta \cdot \text{task\_priority}_i
+ \gamma \cdot \text{braking\_risk}_i
\]

The AMR with higher risk is planned earlier.

## V0 Pipeline

```text
Conflict Cluster
    ↓
Sort AMRs by TTC / risk priority
    ↓
For each AMR in priority order:
    1. Build single-AMR observation
    2. Use existing single-agent policy to propose action
    3. Generate backup candidate actions
    4. Roll out candidate trajectories
    5. Check against worker tube, static map, and previous reservations
    6. Accept the lowest-cost safe trajectory
    7. Add accepted trajectory to reservation table
    ↓
Dispatch local commands
```

## V0 Action Proposal

For AMR \(i\), the existing single-agent policy gives:

\[
a_i^0 = \pi_{\theta}^{single}(o_i)
\]

where the action can be:

\[
a_i^0 = (\text{goal\_fwd}_i, \text{goal\_lat}_i, \text{speed\_scale}_i)
\]

The planner should not rely on only one action. It should build a small candidate action set:

\[
\mathcal{A}_i =
\{a_i^0, \text{slow}, \text{stop}, \text{left shift}, \text{right shift}, \text{shorter forward}\}
\]

Each candidate action is rolled out into a short-horizon trajectory:

\[
\tau_i = \{x_i(t), y_i(t), \theta_i(t), v_i(t)\}_{t=0}^{T}
\]

## V0 Candidate Cost

Each candidate trajectory can be scored by:

\[
J_i =
w_1 J_{worker}
+ w_2 J_{amr}
+ w_3 J_{route}
+ w_4 J_{smooth}
+ w_5 J_{delay}
\]

where:

- \(J_{worker}\): penalty for getting close to worker occupied tube;
- \(J_{amr}\): penalty for getting close to reserved AMR trajectories;
- \(J_{route}\): penalty for deviating from the global path;
- \(J_{smooth}\): penalty for sharp steering, braking, or jerk;
- \(J_{delay}\): penalty for excessive slowing or stopping.

The selected trajectory is:

\[
\tau_i^* = \arg\min_{\tau_i \in \mathcal{T}_i} J_i
\]

subject to all hard safety constraints.

## V0 Strengths

V0 is useful because:

- it reuses the existing single-agent policy;
- it works naturally with variable \(N\);
- it is easy to debug;
- it is real-time friendly;
- it provides a strong baseline for the paper;
- its behavior is interpretable through TTC priority order.

## V0 Limits

V0 has the following limits:

- it is order-dependent;
- it does not fully reason over all AMR actions jointly;
- early AMRs may take space that later AMRs need;
- it may be conservative and choose STOP when candidates fail the shield.

These limits motivate V1.

---

# V1: Attention-Based Shared Policy for Variable-N Multi-AMR Replanning

## V1 Goal

V1 replaces the sequential action proposal in V0 with a joint neural policy.

V1 receives all AMRs in the conflict cluster at the same time:

\[
(o_1,o_2,\ldots,o_N)
\]

and outputs one action per AMR:

\[
(a_1,a_2,\ldots,a_N)
\]

The output action for AMR \(i\) is:

\[
a_i = (\text{goal\_fwd}_i, \text{goal\_lat}_i, \text{speed\_scale}_i)
\]

where:

- `goal_fwd` is the forward distance along the global path;
- `goal_lat` is the lateral offset from the global path;
- `speed_scale` is a normalized speed multiplier in \([0,1]\).

V1 is not the final safety layer. It only generates a joint action proposal. All trajectories generated from V1 actions must still pass the same space-time reservation safety shield.

The core idea is:

```text
V1 attention policy proposes joint actions.
The safety shield filters executable trajectories.
```

---

# V1 Full Structure

```text
                         Step E: V1 Multi-AMR Local Replanning

┌────────────────────────────────────────────────────────────────────┐
│                         Conflict Cluster                           │
│                                                                    │
│   AMR_1, AMR_2, ..., AMR_N                                          │
│   Worker predicted occupied tube W(t)                               │
│   Static map / local obstacles                                      │
│   Global route reference for each AMR                                │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Per-AMR Observation Builder                      │
│                                                                    │
│   For each AMR i:                                                   │
│                                                                    │
│   o_i = [                                                           │
│       ego state,                                                    │
│       route-relative state,                                         │
│       local goal feature,                                           │
│       route preview feature,                                        │
│       worker tube feature,                                          │
│       TTC / priority feature                                        │
│   ]                                                                 │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                        Shared Token Encoder                         │
│                                                                    │
│   z_i = φ_enc(o_i)                                                  │
│                                                                    │
│   Same encoder φ_enc is used for every AMR.                          │
│   Output: token z_i ∈ R^d                                            │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Self-Attention Interaction                     │
│                                                                    │
│   Input tokens: [z_1, z_2, ..., z_N]                                 │
│                                                                    │
│   h_1, h_2, ..., h_N = TransformerEncoder([z_1, ..., z_N], mask)     │
│                                                                    │
│   Each AMR token attends to other AMR tokens in the same cluster.    │
│   The mask ignores padded agents if batching is used.                │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Shared Actor Head                           │
│                                                                    │
│   For each AMR i:                                                   │
│                                                                    │
│   a_i = ψ_actor(h_i)                                                 │
│                                                                    │
│   a_i = (goal_fwd_i, goal_lat_i, speed_scale_i)                      │
│                                                                    │
│   Same actor head ψ_actor is used for every AMR.                     │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Candidate Trajectory Rollout                     │
│                                                                    │
│   Convert each action a_i into a short-horizon trajectory:           │
│                                                                    │
│   τ_i = rollout(amr_i, a_i, H, dt)                                   │
│                                                                    │
│   Optionally also generate backup candidates around a_i.             │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Space-Time Reservation Shield                    │
│                                                                    │
│   Check:                                                            │
│   1. AMR-worker collision with W(t)                                  │
│   2. AMR-AMR collision among proposed trajectories                   │
│   3. static obstacle collision                                       │
│   4. route deviation and kinematic feasibility                       │
│                                                                    │
│   If safe: accept trajectory.                                        │
│   If unsafe: choose nearest safe backup candidate.                   │
│   If no safe candidate exists: STOP.                                 │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Command Dispatch                            │
│                                                                    │
│   For each AMR i:                                                   │
│                                                                    │
│   output local waypoints, target speed, speed limit, or stop command │
└────────────────────────────────────────────────────────────────────┘
```

---

# V1 Observation Design

For each AMR \(i\), build a fixed-dimensional observation vector:

\[
o_i \in \mathbb{R}^{D}
\]

The observation should use route-relative and local-frame features when possible. This makes the policy less tied to absolute map coordinates.

## Ego State Feature

Recommended ego feature:

\[
s_i^{route} =
[s_i^{long}, s_i^{lat}, \theta_i^{rel}, v_i, a_i, \omega_i]
\]

where:

- \(s_i^{long}\): projection progress along the global path;
- \(s_i^{lat}\): lateral offset from the global path;
- \(\theta_i^{rel}\): heading angle relative to the path tangent;
- \(v_i\): current speed;
- \(a_i\): current acceleration;
- \(\omega_i\): angular velocity.

## Local Goal Feature

Represent the local goal in the path frame:

\[
g_i = [\Delta s_i^{goal}, \Delta l_i^{goal}]
\]

where:

- \(\Delta s_i^{goal}\): longitudinal distance to the local goal;
- \(\Delta l_i^{goal}\): lateral distance to the local goal.

## Route Preview Feature

Extract the next \(M\) waypoints on the global path:

\[
P_i = [p_i^1, p_i^2, \ldots, p_i^M]
\]

Each waypoint can be represented in the AMR local frame:

\[
p_i^m = [\Delta x_i^m, \Delta y_i^m, \Delta \theta_i^m]
\]

Flatten them into:

\[
\text{route\_feature}_i \in \mathbb{R}^{3M}
\]

Suggested first version:

```python
M = 10
route_feature_dim = 30
```

## Worker Tube Feature

The worker prediction gives a future occupied tube:

\[
\mathcal{W}(t), \quad t = 0,\ldots,H
\]

For each AMR, extract compact risk features instead of passing a full occupancy grid.

For one worker:

\[
w_i =
[
 d_{min}^{worker},
 t_{min}^{worker},
 \Delta x_{worker}^{risk},
 \Delta y_{worker}^{risk},
 \sigma_x,
 \sigma_y,
 \text{worker\_risk\_score}
]
\]

For multiple workers, use the top-\(K\) most risky workers and flatten their features.

Suggested first version:

```python
max_workers = 3
worker_feature_dim_per_worker = 7
worker_feature_dim = 21
```

## Priority and Risk Feature

Use:

\[
q_i =
[
\text{TTC}_i,
1/(\text{TTC}_i + \epsilon),
\text{task\_priority}_i,
\text{braking\_risk}_i,
\text{affected\_flag}_i
]
\]

All features should be normalized before being fed to the neural network.

Example:

\[
\text{TTC}_i^{norm} = \text{clip}(\text{TTC}_i/T_{max}, 0, 1)
\]

## Final Per-AMR Observation

The final per-AMR observation is:

\[
o_i =
[
s_i^{route},
g_i,
\text{route\_feature}_i,
\text{worker\_feature}_i,
q_i
]
\]

Implementation interface:

```python
obs_i = {
    "ego": ego_feature,              # shape: [ego_dim]
    "goal": goal_feature,            # shape: [goal_dim]
    "route": route_feature,          # shape: [route_dim]
    "worker": worker_feature,        # shape: [worker_dim]
    "priority": priority_feature,    # shape: [priority_dim]
}

obs_vec_i = concat([ego, goal, route, worker, priority])
```

---

# V1 Network Architecture

## High-Level Model

The V1 policy is a shared-weight attention network:

\[
z_i = \phi_{enc}(o_i)
\]

\[
h_1,\ldots,h_N = \text{TransformerEncoder}(z_1,\ldots,z_N)
\]

\[
a_i = \psi_{actor}(h_i)
\]

If PPO-style training is used, add a centralized critic:

\[
V(\mathcal{C}) = \psi_{critic}(\text{pool}(h_1,\ldots,h_N))
\]

## Input and Mask

The policy should support padded batches.

Input:

```python
obs.shape = [B, N_max, obs_dim]
mask.shape = [B, N_max]
```

where:

- `B` is batch size;
- `N_max` is the maximum number of agents, initially 4;
- `mask[b, i] = 1` means the AMR is valid;
- `mask[b, i] = 0` means the slot is padding.

PyTorch `TransformerEncoder` uses `src_key_padding_mask`, where `True` means ignored. Therefore:

```python
src_key_padding_mask = (mask == 0)
```

## Actor Output

Output:

```python
action_mean.shape = [B, N_max, action_dim]
value.shape = [B, 1]
```

Use:

```python
action_dim = 3
```

The raw action is decoded as:

```python
goal_fwd = max_forward_dist * sigmoid(raw_action[..., 0])
goal_lat = max_lateral_offset * tanh(raw_action[..., 1])
speed_scale = sigmoid(raw_action[..., 2])
```

Action ranges:

```python
goal_fwd     in [0.0, max_forward_dist]
goal_lat     in [-max_lateral_offset, max_lateral_offset]
speed_scale  in [0.0, 1.0]
```

## Critic Output

Use masked mean pooling:

```python
cluster_embedding = masked_mean(hidden, mask)
value = critic_head(cluster_embedding)
```

The value is cluster-level:

```python
value.shape = [B, 1]
```

Cluster-level value is preferred for the first implementation because Step E is centralized and the reward can be cluster-level.

---

# V1 Network Structure Diagram

```text
             V1: Attention-Based Shared Policy

Input Conflict Cluster:
    C = {AMR_1, AMR_2, ..., AMR_N}, N <= 4

For each AMR i:

    ┌─────────────────────────────────────────┐
    │ Observation o_i                          │
    │                                         │
    │ ego state                               │
    │ route-relative state                    │
    │ local goal                              │
    │ route preview                           │
    │ worker tube feature                     │
    │ TTC / priority feature                  │
    └─────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │ Shared MLP Encoder φ_enc                 │
    │                                         │
    │ z_i = φ_enc(o_i)                         │
    └─────────────────────────────────────────┘
                         │
                         ▼

All AMR tokens:

    ┌─────────────────────────────────────────┐
    │ Token Set                               │
    │                                         │
    │ [z_1, z_2, ..., z_N]                     │
    │                                         │
    │ padded to [z_1, ..., z_N, pad, pad]      │
    │ with attention mask                     │
    └─────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │ Transformer Encoder                      │
    │                                         │
    │ self-attention over AMR tokens           │
    │                                         │
    │ h_1, h_2, ..., h_N                       │
    └─────────────────────────────────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Shared Actor Head          │   │ Centralized Critic Head    │
│                           │   │                           │
│ a_i = ψ_actor(h_i)         │   │ V = ψ_critic(pool(h_i))    │
│                           │   │                           │
│ output per-AMR action      │   │ output cluster value       │
└───────────────────────────┘   └───────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Joint Action Proposal                    │
│                                         │
│ A = {a_1, a_2, ..., a_N}                 │
│                                         │
│ a_i = (goal_fwd_i, goal_lat_i,           │
│        speed_scale_i)                    │
└─────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Trajectory Rollout                       │
│                                         │
│ τ_i = rollout(amr_i, a_i, H, dt)         │
└─────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Space-Time Reservation Shield            │
│                                         │
│ check worker tube                        │
│ check AMR-AMR collision                  │
│ check static obstacles                   │
│ check dynamic feasibility                │
│                                         │
│ safe: execute                            │
│ unsafe: backup candidate or STOP         │
└─────────────────────────────────────────┘
```

---

# Suggested File Structure

Implement the first version using the following structure:

```text
step_e_v1/
│
├── config.py
│   └── V1Config
│
├── observation_builder.py
│   └── build_cluster_observation(...)
│   └── build_amr_observation(...)
│   └── extract_route_feature(...)
│   └── extract_worker_tube_feature(...)
│
├── attention_policy.py
│   └── MultiAMRAttentionPolicy
│   └── SharedTokenEncoder
│   └── TransformerInteractionModule
│   └── ActorHead
│   └── CriticHead
│
├── action_decoder.py
│   └── decode_action(...)
│   └── action_to_local_goal(...)
│
├── trajectory_rollout.py
│   └── rollout_action(...)
│   └── generate_backup_candidates(...)
│
├── safety_shield.py
│   └── check_worker_collision(...)
│   └── check_amr_amr_collision(...)
│   └── check_static_obstacle_collision(...)
│   └── select_safe_trajectory(...)
│   └── select_safe_joint_trajectories(...)
│
├── v0_planner.py
│   └── V0SequentialReplanner
│
├── v1_planner.py
│   └── V1AttentionReplanner
│
└── train/
    └── ppo_trainer.py
```

The first implementation can skip training and only implement the policy forward pass, planner interface, rollout, and shield. Training can be added later.

---

# Core Config

```python
from dataclasses import dataclass

@dataclass
class V1Config:
    max_agents: int = 4
    obs_dim: int = 64
    token_dim: int = 128
    action_dim: int = 3

    num_attention_layers: int = 2
    num_attention_heads: int = 4
    transformer_ff_dim: int = 256
    dropout: float = 0.1

    max_forward_dist: float = 3.0
    max_lateral_offset: float = 1.0

    horizon_sec: float = 3.0
    dt: float = 0.1

    amr_radius: float = 0.45
    worker_safety_radius: float = 0.8
    min_amr_distance: float = 0.9

    max_workers: int = 3
    route_preview_points: int = 10

    use_backup_candidates: bool = True
```

Note: `obs_dim` must match the final feature vector length produced by `observation_builder.py`. Adjust it after the observation fields are fixed.

---

# Core Network Class

Implement the policy in PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedTokenEncoder(nn.Module):
    def __init__(self, obs_dim: int, token_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TransformerInteractionModule(nn.Module):
    def __init__(self, token_dim: int, num_layers: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = mask == 0
        return self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)


class ActorHead(nn.Module):
    def __init__(self, token_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, action_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class CriticHead(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, 1),
        )

    def forward(self, cluster_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(cluster_embedding)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.float().unsqueeze(-1)
    hidden_sum = (hidden * mask_float).sum(dim=1)
    denom = mask_float.sum(dim=1).clamp(min=1.0)
    return hidden_sum / denom


def decode_action(raw_action: torch.Tensor, max_forward_dist: float, max_lateral_offset: float) -> torch.Tensor:
    goal_fwd = max_forward_dist * torch.sigmoid(raw_action[..., 0:1])
    goal_lat = max_lateral_offset * torch.tanh(raw_action[..., 1:2])
    speed_scale = torch.sigmoid(raw_action[..., 2:3])
    return torch.cat([goal_fwd, goal_lat, speed_scale], dim=-1)


class MultiAMRAttentionPolicy(nn.Module):
    """
    Attention-based shared policy for variable-N multi-AMR local replanning.

    Input:
        obs:  Tensor [B, N_max, obs_dim]
        mask: Tensor [B, N_max], 1 for valid AMR, 0 for padded AMR

    Output:
        action: Tensor [B, N_max, action_dim]
        value:  Tensor [B, 1]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = SharedTokenEncoder(
            obs_dim=config.obs_dim,
            token_dim=config.token_dim,
        )

        self.transformer = TransformerInteractionModule(
            token_dim=config.token_dim,
            num_layers=config.num_attention_layers,
            num_heads=config.num_attention_heads,
            ff_dim=config.transformer_ff_dim,
            dropout=config.dropout,
        )

        self.actor_head = ActorHead(
            token_dim=config.token_dim,
            action_dim=config.action_dim,
        )

        self.critic_head = CriticHead(
            token_dim=config.token_dim,
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        tokens = self.encoder(obs)
        hidden = self.transformer(tokens, mask)

        raw_action = self.actor_head(hidden)
        action = decode_action(
            raw_action,
            max_forward_dist=self.config.max_forward_dist,
            max_lateral_offset=self.config.max_lateral_offset,
        )

        cluster_embedding = masked_mean(hidden, mask)
        value = self.critic_head(cluster_embedding)

        return action, value
```

---

# V1 Planner Runtime Flow

The online planner should use the following logic:

```text
1. Receive conflict cluster.
2. Build per-AMR observation.
3. Pad AMR tokens to max_agents.
4. Run attention policy.
5. Decode bounded actions.
6. Roll out one trajectory per AMR.
7. Check the joint trajectories with the safety shield.
8. If unsafe, try backup candidates.
9. If no safe candidate exists, output STOP.
10. Dispatch final commands.
```

Pseudo-code:

```python
class V1AttentionReplanner:
    def __init__(self, policy, config, device="cpu"):
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.policy.eval()

    def plan(self, cluster, worker_predictions, map_data, dt=None):
        if dt is None:
            dt = self.config.dt

        obs, mask, amr_ids = build_cluster_observation(
            cluster=cluster,
            worker_predictions=worker_predictions,
            map_data=map_data,
            config=self.config,
        )

        obs = obs.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            actions, value = self.policy(obs, mask)

        action_dict = {}
        for k, amr_id in enumerate(amr_ids):
            action_dict[amr_id] = actions[0, k].detach().cpu().numpy()

        candidate_trajs = {}
        for amr in cluster:
            action = action_dict[amr.id]
            candidate_trajs[amr.id] = rollout_action(
                amr=amr,
                action=action,
                horizon_sec=self.config.horizon_sec,
                dt=dt,
                map_data=map_data,
            )

        safe_trajs = select_safe_joint_trajectories(
            cluster=cluster,
            candidate_trajs=candidate_trajs,
            worker_predictions=worker_predictions,
            map_data=map_data,
            config=self.config,
        )

        commands = {}
        for amr in cluster:
            commands[amr.id] = trajectory_to_command(
                amr=amr,
                trajectory=safe_trajs[amr.id],
            )

        return commands
```

---

# Trajectory Rollout

For each action:

\[
a_i = (\text{goal\_fwd}, \text{goal\_lat}, \text{speed\_scale})
\]

convert it into a local target:

```text
target = route_projection
       + goal_fwd * path_tangent
       + goal_lat * path_normal
```

Then generate a short-horizon trajectory:

\[
\tau_i = \text{rollout}(amr_i, a_i, H, dt)
\]

The trajectory output should use this format:

```python
@dataclass
class Trajectory:
    times: np.ndarray       # shape: [T]
    positions: np.ndarray   # shape: [T, 2]
    headings: np.ndarray    # shape: [T]
    speeds: np.ndarray      # shape: [T]
```

First implementation options:

- spline interpolation from current pose to local target;
- pure-pursuit style rollout;
- constant acceleration speed profile;
- simple unicycle rollout.

The first version does not need a perfect trajectory generator. It only needs consistent short-horizon rollout for safety checking.

---

# Space-Time Reservation Safety Shield

The safety shield is shared by V0 and V1.

For each trajectory:

\[
\tau_i = \{p_i(t), \theta_i(t), v_i(t)\}_{t=0}^{T}
\]

where:

\[
p_i(t) = [x_i(t), y_i(t)]
\]

The shield checks four constraints.

## Worker Safety

If the worker tube is represented by circles:

\[
\|p_i(t) - p_w(t)\| > r_{amr} + r_{worker} + margin
\]

for every AMR \(i\), worker \(w\), and time step \(t\).

If the worker tube is represented by ellipses, check whether the AMR footprint enters the ellipse.

## AMR-AMR Safety

For every pair \((i,j)\):

\[
\|p_i(t) - p_j(t)\| > d_{min}
\]

for every time step \(t\).

## Static Obstacle Safety

Each AMR footprint must stay outside inflated static obstacles.

## Dynamic Feasibility

Check basic limits:

- maximum speed;
- maximum acceleration;
- maximum angular velocity;
- maximum curvature if available.

## Fallback Logic

If the V1 proposal is unsafe:

```text
V1 proposal unsafe
    ↓
generate backup candidates around V1 action
    ↓
select the lowest-cost safe candidate
    ↓
if no safe candidate exists, STOP
```

Backup actions:

```python
backup_actions = [
    original_action,
    slow_down_action,
    stop_action,
    small_left_shift_action,
    small_right_shift_action,
    shorter_forward_action,
]
```

The STOP command must always be available as the final fallback.

---

# Training Plan for V1

The first implementation does not need full training. However, the policy should return both action and value so PPO-style training can be added.

Use:

```text
shared actor + centralized critic + PPO update
```

The training loop should store transitions:

```python
transition = {
    "obs": obs,                    # [N_max, obs_dim]
    "mask": mask,                  # [N_max]
    "action": action,              # [N_max, action_dim]
    "log_prob": log_prob,          # [N_max] or scalar
    "reward": reward,              # scalar cluster reward
    "value": value,                # scalar
    "done": done,
    "next_obs": next_obs,
    "next_mask": next_mask,
}
```

## Curriculum Training

Use curriculum training:

```text
Stage 1: N=1 warm-start
    Learn to match the existing single-agent policy.

Stage 2: N=2 simple interaction
    Learn yielding, slowing, and small lateral shifts.

Stage 3: N=3
    Learn group coordination.

Stage 4: N=4
    Handle full conflict clusters.

Stage 5: mixed-N training
    Randomly sample N from {1,2,3,4}.
```

Stage 1 can use behavior cloning:

\[
\mathcal{L}_{BC} = \|a_i^{V1} - a_i^{single}\|^2
\]

Later stages can use PPO.

## Reward Design

Use a cluster-level reward:

\[
r =
-r_{collision}
-r_{worker\_risk}
-r_{amr\_risk}
-r_{delay}
-r_{route\_deviation}
-r_{jerk}
-r_{shield}
\]

where:

- \(r_{collision}\): large penalty for worker or AMR collision;
- \(r_{worker\_risk}\): penalty for getting close to worker tube;
- \(r_{amr\_risk}\): penalty for AMR-AMR closeness;
- \(r_{delay}\): penalty for excessive slowing or stopping;
- \(r_{route\_deviation}\): penalty for moving far from global path;
- \(r_{jerk}\): penalty for non-smooth control;
- \(r_{shield}\): penalty if the safety shield modifies the policy output.

The shield penalty is important. Without it, the policy may keep outputting unsafe actions and rely on the shield to fix them.

Possible shield penalty:

\[
r_{shield} =
\lambda_{shield}
\cdot
\mathbb{1}[\text{shield modified action}]
\]

or:

\[
r_{shield} =
\lambda_{shield}
\cdot
\|a_{raw} - a_{executed}\|^2
\]

---

# Implementation Requirements for Claude Code

Implement V1 for Step E: an attention-based shared policy for variable-N multi-AMR local replanning.

The planner receives a conflict cluster with \(N\) AMRs, where \(N \leq 4\). It must build per-AMR observations, run a shared-weight attention policy, output one action per AMR, roll out short-horizon trajectories, and then pass all trajectories through a space-time reservation safety shield.

## Required Modules

### 1. Per-AMR Observation Builder

Implement:

```python
build_cluster_observation(cluster, worker_predictions, map_data, config)
build_amr_observation(amr, cluster, worker_predictions, map_data, config)
extract_route_feature(amr, map_data, config)
extract_worker_tube_feature(amr, worker_predictions, config)
```

Output:

```python
obs.shape = [1, config.max_agents, config.obs_dim]
mask.shape = [1, config.max_agents]
amr_ids = list of valid AMR ids
```

### 2. Shared Token Encoder

Implement:

```python
z_i = MLP(o_i)
```

The same MLP is used for every AMR.

### 3. Transformer Interaction Module

Run self-attention over valid AMR tokens.

Use padding mask so padded AMRs are ignored.

### 4. Actor Head

For each AMR token \(h_i\), output:

```python
raw_action_i = [raw_goal_fwd, raw_goal_lat, raw_speed_scale]
```

Decode:

```python
goal_fwd = max_forward_dist * sigmoid(raw_goal_fwd)
goal_lat = max_lateral_offset * tanh(raw_goal_lat)
speed_scale = sigmoid(raw_speed_scale)
```

### 5. Critic Head

Use masked mean pooling over hidden AMR tokens and output one cluster-level value.

### 6. Rollout

Convert each action into a local target:

```text
target = route_projection
       + goal_fwd * path_tangent
       + goal_lat * path_normal
```

Generate a short-horizon trajectory with:

```python
horizon_sec = 3.0
dt = 0.1
```

### 7. Safety Shield

Check:

- AMR-worker collision using worker predicted occupied tube;
- AMR-AMR collision among all proposed trajectories;
- static obstacle collision;
- dynamic feasibility.

If the proposed trajectory is unsafe, generate backup candidates:

```text
original, slow down, stop, small left shift, small right shift, shorter forward
```

Select the lowest-cost safe candidate.

If no safe candidate exists, output STOP.

### 8. Planner Interface

Create:

```python
class V1AttentionReplanner:
    def plan(self, cluster, worker_predictions, map_data, dt) -> dict:
        ...
```

Return:

```python
commands = {
    amr_id: command,
    ...
}
```

Each command should include at least:

```python
command = {
    "mode": "TRACK" or "STOP",
    "waypoints": ...,
    "target_speed": ...,
    "speed_limit": ...,
}
```

---

# Final Instruction

Implement the first version as a clean, modular Python package. Use PyTorch for the neural network. Training code is optional in the first version, but the policy must return both actions and a cluster-level value so PPO-style training can be added later.

The implementation should prioritize:

1. clear module boundaries;
2. correct tensor shapes;
3. mask support for variable \(N\);
4. safety shield integration;
5. easy replacement of placeholder functions with real simulator functions later.
