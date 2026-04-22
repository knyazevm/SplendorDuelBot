"""
network.py — Policy + Value MLP for Splendor Duel.

Architecture:
    obs[519] → Linear(256) → ReLU → Linear(128) → ReLU
                                        ├── policy_head → logits[265]
                                        └── value_head  → scalar
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from splendor_duel.env import OBS_SIZE, N_ACTIONS


class SplendorNetwork(nn.Module):

    def __init__(self, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden2, N_ACTIONS)
        self.value_head = nn.Linear(hidden2, 1)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        return self.policy_head(features), self.value_head(features)

    def get_action(
            self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Sample one action (single obs, no grad)."""
        with torch.no_grad():
            logits, value = self.forward(obs.unsqueeze(0))
            logits = logits.squeeze(0)
            logits = logits.masked_fill(~mask, float('-inf'))
            dist = Categorical(logits=logits)
            action = logits.argmax().item() if deterministic else dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        return action, log_prob, value.item()

    def evaluate_actions(
            self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch evaluation for PPO loss. Returns (log_probs, values, entropy)."""
        logits, values = self.forward(obs)
        values = values.squeeze(-1)
        logits = logits.masked_fill(~masks, float('-inf'))
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()
