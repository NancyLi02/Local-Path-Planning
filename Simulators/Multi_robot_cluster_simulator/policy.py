"""Attention policy for the multi-robot cluster env (variable AMR count).

Per-AMR token -> shared encoder + Transformer self-attention -> shared actor
head (2D tanh-Gaussian action = body-frame local goal) + centralized critic.
Separate actor/critic trunks for stable PPO. Same design as the deployment
policy, kept self-contained in this training package.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ACT_DIM = 2
_LSMIN, _LSMAX = -5.0, 1.0


def _trunk(obs_dim, hidden, heads, layers):
    enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU())
    lyr = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,
                                     dim_feedforward=hidden * 2, dropout=0.0,
                                     batch_first=True, activation="gelu")
    return enc, nn.TransformerEncoder(lyr, num_layers=layers)


def _run(enc, attn, obs, mask):
    h = enc(obs)
    pad = ~mask
    allpad = pad.all(dim=1)
    if allpad.any():
        pad = pad.clone(); pad[allpad] = False
    return attn(h, src_key_padding_mask=pad)


class ClusterPolicy(nn.Module):
    def __init__(self, obs_dim, hidden=96, heads=4, layers=2):
        super().__init__()
        self.a_enc, self.a_attn = _trunk(obs_dim, hidden, heads, layers)
        self.c_enc, self.c_attn = _trunk(obs_dim, hidden, heads, layers)
        self.mean = nn.Linear(hidden, ACT_DIM)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, mask):
        ha = _run(self.a_enc, self.a_attn, obs, mask)
        mean = self.mean(ha)
        std = self.log_std.clamp(_LSMIN, _LSMAX).exp()
        hc = _run(self.c_enc, self.c_attn, obs, mask)
        m = mask.unsqueeze(-1).float()
        pooled = (hc * m).sum(1) / m.sum(1).clamp(min=1.0)
        value = self.critic(pooled).squeeze(-1)
        return mean, std, value

    def _logp(self, mean, std, action, mask):
        a = action.clamp(-1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(a)
        base = -0.5 * (((u - mean) / std) ** 2) - std.log() - 0.5 * torch.log(
            torch.tensor(2.0 * torch.pi, device=mean.device))
        corr = torch.log(1.0 - a ** 2 + 1e-6)
        return ((base - corr).sum(-1) * mask.float()).sum(-1)

    @torch.no_grad()
    def act(self, obs, mask, deterministic=False):
        mean, std, value = self.forward(obs, mask)
        u = mean if deterministic else mean + std * torch.randn_like(mean)
        action = torch.tanh(u)
        return action, self._logp(mean, std, action, mask), value

    def evaluate(self, obs, mask, action):
        mean, std, value = self.forward(obs, mask)
        logp = self._logp(mean, std, action, mask)
        ent = (0.5 + 0.5 * torch.log(torch.tensor(2.0 * torch.pi)) + std.log()).sum()
        return logp, ent.expand(obs.shape[0]), value
