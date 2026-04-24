"""
network.py — Stabilized Policy + Value MLP for Splendor Duel.

Uses LayerNorm for training stability and clamped logits
to prevent NaN from gradient explosion.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from splendor_duel.env import OBS_SIZE, N_ACTIONS


class SplendorNetwork(nn.Module):

    def __init__(self, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        # Shared backbone with LayerNorm for stability
        self.shared = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden2, N_ACTIONS)
        self.value_head = nn.Linear(hidden2, 1)

        # Small init for policy (near-uniform initial distribution)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        logits = self.policy_head(features).clamp(-15, 15)
        value = self.value_head(features)
        return logits, value

    def get_action(
            self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Sample one action. Fully crash-proof."""
        # Safety: if no legal actions at all, return action 0
        legal = torch.where(mask)[0]
        if len(legal) == 0:
            return 0, 0.0, 0.0

        with torch.no_grad():
            logits, value = self.forward(obs.unsqueeze(0))
            logits = logits.squeeze(0)
            val = value.item()

            # Fallback for corrupted network output
            if not torch.isfinite(logits).all() or not math.isfinite(val):
                action = legal[torch.randint(len(legal), (1,))].item()
                return action, 0.0, 0.0

            # Safe masked softmax: -1e9 instead of -inf
            logits[~mask] = -1e9
            logits = logits - logits.max()
            exp_logits = torch.exp(logits)
            exp_logits[~mask] = 0.0
            total = exp_logits.sum()

            if total < 1e-30:
                action = legal[torch.randint(len(legal), (1,))].item()
                return action, 0.0, 0.0

            probs = exp_logits / total

            if deterministic:
                action = probs.argmax().item()
            else:
                action = torch.multinomial(probs, 1).item()

            log_prob = torch.log(probs[action] + 1e-10).item()
            return action, log_prob, val

    def evaluate_actions(
            self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch evaluation for PPO loss. NaN-safe."""
        logits, values = self.forward(obs)
        values = values.squeeze(-1)

        # Safe masked softmax: use large negative instead of -inf
        logits = logits.masked_fill(~masks, -1e9)
        logits = logits - logits.max(dim=-1, keepdim=True).values  # stability
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * masks.float()  # zero out illegal
        probs = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + 1e-10)
        probs = probs + 1e-10  # avoid log(0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = Categorical(probs=probs, validate_args=False)
        return dist.log_prob(actions), values, dist.entropy()
