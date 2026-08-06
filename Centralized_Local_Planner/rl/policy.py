"""V1 attention-based centralized shared policy (PyTorch).

A permutation-equivariant fleet policy:
- each AMR's features are encoded by a SHARED MLP (one set of weights for all
  AMRs -> handles a variable number of agents),
- a small Transformer self-attention block lets the AMRs attend to each other
  (centralized coordination; inactive AMRs are masked out),
- a SHARED actor head outputs a Beta(alpha, beta) over each AMR's desired speed
  factor in (0, 1),
- a centralized critic pools the (masked) agent embeddings into a single fleet
  state value.

The sampled factor is fed to the FleetEnv, which projects it through the V0
space-time safety shield, so the policy only ever shapes efficiency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class AttentionPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.attn = nn.TransformerEncoder(enc, num_layers=layers)
        self.actor = nn.Linear(hidden, 2)          # -> Beta(alpha, beta)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )
        # Warm-start toward V0: Beta mean ~0.8 which, combined with the env's
        # action slack (x1.25), proposes ~full max-safe speed -> begins close to
        # the strong greedy baseline and learns to *reduce* speed only where it
        # helps (clearance / smoothness). Keep default (non-zero) actor weights
        # so AMRs can be differentiated by the attention context.
        with torch.no_grad():
            self.actor.weight.mul_(0.3)
            self.actor.bias.copy_(torch.tensor([1.6, -5.0]))   # alpha~4, beta~1 -> mean ~0.80

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """obs (B,N,D), mask (B,N) bool (True=active). Returns alpha,beta,value."""
        h = self.encoder(obs)
        pad = ~mask                                  # True = ignore in attention
        all_pad = pad.all(dim=1)                     # rows with zero active AMRs
        if all_pad.any():
            pad = pad.clone()
            pad[all_pad] = False                     # avoid NaN; outputs masked later
        h = self.attn(h, src_key_padding_mask=pad)
        ab = F.softplus(self.actor(h)) + 1.0         # alpha,beta > 1 (unimodal Beta)
        alpha, beta = ab[..., 0], ab[..., 1]
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        value = self.critic(pooled).squeeze(-1)
        return alpha, beta, value

    # -- helpers ----------------------------------------------------------
    def dist(self, obs, mask):
        alpha, beta, value = self.forward(obs, mask)
        return Beta(alpha, beta), value

    @torch.no_grad()
    def act(self, obs, mask, deterministic: bool = False):
        d, value = self.dist(obs, mask)
        if deterministic:
            action = d.mean
        else:
            action = d.sample()
        action = action.clamp(1e-4, 1 - 1e-4)
        logp = (d.log_prob(action) * mask.float()).sum(dim=-1)   # joint over active
        return action, logp, value

    def evaluate(self, obs, mask, action):
        d, value = self.dist(obs, mask)
        action = action.clamp(1e-4, 1 - 1e-4)
        mf = mask.float()
        logp = (d.log_prob(action) * mf).sum(dim=-1)
        entropy = (d.entropy() * mf).sum(dim=-1) / mf.sum(dim=-1).clamp(min=1.0)
        return logp, entropy, value
