"""V1 attention-based shared policy for variable-N multi-AMR replanning.

    z_i = phi_enc(o_i)                       shared per-AMR MLP encoder
    h_1..h_N = TransformerEncoder(z_1..z_N)  self-attention over the cluster
    a_i = psi_actor(h_i)                     shared actor head  -> per-AMR action
    V   = psi_critic(pool(h_1..h_N))         centralized cluster-level critic

The module is permutation-equivariant and mask-aware, so one set of weights
serves clusters of any size up to ``config.max_agents``.

``forward`` returns the DECODED action and the cluster value (specification
interface). For PPO the raw pre-decode Gaussian is exposed through
``act`` / ``evaluate_actions``, which keep the log-probability in raw space
(the decode is a fixed invertible squashing, so raw-space PPO is exact).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .action_decoder import decode_action

_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 1.0


class SharedTokenEncoder(nn.Module):
    def __init__(self, obs_dim: int, token_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, token_dim), nn.ReLU(),
            nn.Linear(token_dim, token_dim), nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TransformerInteractionModule(nn.Module):
    def __init__(self, token_dim: int, num_layers: int, num_heads: int,
                 ff_dim: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = mask == 0
        # A fully padded row would make softmax produce NaNs; unmask it (its
        # outputs are discarded by the caller's mask anyway).
        allpad = src_key_padding_mask.all(dim=1)
        if bool(allpad.any()):
            src_key_padding_mask = src_key_padding_mask.clone()
            src_key_padding_mask[allpad] = False
        return self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)


class ActorHead(nn.Module):
    def __init__(self, token_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.ReLU(),
            nn.Linear(token_dim, action_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class CriticHead(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.ReLU(),
            nn.Linear(token_dim, 1),
        )

    def forward(self, cluster_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(cluster_embedding)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.float().unsqueeze(-1)
    hidden_sum = (hidden * mask_float).sum(dim=1)
    denom = mask_float.sum(dim=1).clamp(min=1.0)
    return hidden_sum / denom


class MultiAMRAttentionPolicy(nn.Module):
    """Attention-based shared policy for variable-N multi-AMR local replanning.

    Input:
        obs:  Tensor [B, N_max, obs_dim]
        mask: Tensor [B, N_max], 1 for valid AMR, 0 for padded AMR

    Output (forward):
        action: Tensor [B, N_max, action_dim]   decoded, bounded
        value:  Tensor [B, 1]                   cluster-level
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = SharedTokenEncoder(config.obs_dim, config.token_dim)
        self.transformer = TransformerInteractionModule(
            token_dim=config.token_dim,
            num_layers=config.num_attention_layers,
            num_heads=config.num_attention_heads,
            ff_dim=config.transformer_ff_dim,
            dropout=config.dropout,
        )
        self.actor_head = ActorHead(config.token_dim, config.action_dim)
        self.critic_head = CriticHead(config.token_dim)
        self.log_std = nn.Parameter(torch.full((config.action_dim,),
                                                getattr(config, "init_log_std", -1.0)))

    # -- shared trunk -------------------------------------------------------
    def _trunk(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.encoder(obs), mask)

    def raw_forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """Pre-decode mean, std and cluster value."""
        hidden = self._trunk(obs, mask)
        raw_mean = self.actor_head(hidden)
        std = self.log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp()
        value = self.critic_head(masked_mean(hidden, mask))
        return raw_mean, std, value

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        raw_mean, _, value = self.raw_forward(obs, mask)
        action = decode_action(raw_mean, self.config.max_forward_dist,
                               self.config.max_lateral_offset)
        return action, value

    # -- PPO ---------------------------------------------------------------
    def _log_prob(self, raw_mean, std, raw_action, mask) -> torch.Tensor:
        var = std ** 2
        lp = -0.5 * (((raw_action - raw_mean) ** 2) / var
                     + 2 * std.log() + math.log(2 * math.pi))
        lp = lp.sum(-1)                                    # per agent
        return (lp * mask.float()).sum(-1)                 # cluster joint log-prob

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
        """Sample a joint action. Returns (decoded, raw, log_prob, value)."""
        raw_mean, std, value = self.raw_forward(obs, mask)
        raw = raw_mean if deterministic else raw_mean + std * torch.randn_like(raw_mean)
        logp = self._log_prob(raw_mean, std, raw, mask)
        decoded = decode_action(raw, self.config.max_forward_dist,
                                self.config.max_lateral_offset)
        return decoded, raw, logp, value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, mask: torch.Tensor,
                         raw_action: torch.Tensor):
        raw_mean, std, value = self.raw_forward(obs, mask)
        logp = self._log_prob(raw_mean, std, raw_action, mask)
        ent_per_dim = 0.5 + 0.5 * math.log(2 * math.pi) + std.log()
        entropy = ent_per_dim.sum().expand(obs.shape[0])
        return logp, entropy, value.squeeze(-1)
