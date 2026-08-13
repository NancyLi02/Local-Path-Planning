"""Attention policy for spatial local replanning (2D continuous local goal).

Permutation-equivariant over the variable set of in-cluster AMRs. Actor and
critic use SEPARATE trunks (each: shared per-AMR encoder + Transformer
self-attention) so value-function updates never disturb the actor -- this makes
BC warm-start + PPO fine-tuning stable. The actor outputs a 2D displacement
(tanh-squashed diagonal Gaussian) = each AMR's local goal; the 2D shield then
projects it to safety, so the policy can be aggressive.

``obs`` is (batch, tokens, obs_dim) and attention runs over the token axis only.
Callers therefore put ONE conflict cluster per batch row -- see
``rl/cluster_policy.py`` -- which keeps clusters from attending to each other
while sharing a single set of weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ACT_DIM = 2
_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 1.0


def _make_trunk(obs_dim, hidden, heads, layers):
    enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU())
    layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,
                                       dim_feedforward=hidden * 2, dropout=0.0,
                                       batch_first=True, activation="gelu")
    attn = nn.TransformerEncoder(layer, num_layers=layers)
    return enc, attn


def _run_trunk(enc, attn, obs, mask):
    h = enc(obs)
    pad = ~mask
    allpad = pad.all(dim=1)
    if allpad.any():
        pad = pad.clone(); pad[allpad] = False
    return attn(h, src_key_padding_mask=pad)


class LocalAttentionPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.a_enc, self.a_attn = _make_trunk(obs_dim, hidden, heads, layers)
        self.c_enc, self.c_attn = _make_trunk(obs_dim, hidden, heads, layers)
        self.mean = nn.Linear(hidden, ACT_DIM)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.5))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1))

    def actor_params(self):
        return (list(self.a_enc.parameters()) + list(self.a_attn.parameters())
                + list(self.mean.parameters()) + [self.log_std])

    def critic_params(self):
        return (list(self.c_enc.parameters()) + list(self.c_attn.parameters())
                + list(self.critic.parameters()))

    def forward(self, obs, mask):
        ha = _run_trunk(self.a_enc, self.a_attn, obs, mask)
        mean = self.mean(ha)
        std = self.log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp()
        hc = _run_trunk(self.c_enc, self.c_attn, obs, mask)
        m = mask.unsqueeze(-1).float()
        pooled = (hc * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        value = self.critic(pooled).squeeze(-1)
        return mean, std, value

    def _logp(self, mean, std, action, mask):
        a = action.clamp(-1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(a)
        base = -0.5 * (((u - mean) / std) ** 2) - std.log() - 0.5 * torch.log(
            torch.tensor(2.0 * torch.pi, device=mean.device))
        corr = torch.log(1.0 - a ** 2 + 1e-6)
        logp_agent = (base - corr).sum(-1)
        return (logp_agent * mask.float()).sum(-1)

    @torch.no_grad()
    def act(self, obs, mask, deterministic=False):
        mean, std, value = self.forward(obs, mask)
        u = mean if deterministic else mean + std * torch.randn_like(mean)
        action = torch.tanh(u)
        logp = self._logp(mean, std, action, mask)
        return action, logp, value

    def evaluate(self, obs, mask, action):
        mean, std, value = self.forward(obs, mask)
        logp = self._logp(mean, std, action, mask)
        ent = (0.5 + 0.5 * torch.log(torch.tensor(2.0 * torch.pi)) + std.log()).sum()
        return logp, ent.expand(obs.shape[0]), value
