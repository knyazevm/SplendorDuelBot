"""
network_az.py — AlphaZero-specific network + checkpoint I/O.

Identical parameters/state_dict to `ppo.network.SplendorNetwork` (so a PPO
checkpoint still warm-starts it), with one change that matters for AZ:

    the value head is squashed through tanh.

Why that matters: AZ value targets are game outcomes in {-1, 0, +1}, and PUCT
balances Q against `c_puct * P * sqrt(N)/(1+N)` assuming Q lives in [-1, 1].
The PPO network's value head is a bare Linear (correct there — PPO regresses
unbounded returns), so an AZ-trained one drifts outside the range. Measured on
the existing `checkpoints_az_tuned/az_80.pt`: 23% of held-out positions were
predicted outside [-1, 1], up to ±1.7. That silently rescales the exploration
term and lets a single bad leaf dominate a subtree.

Checkpoints written here carry `"tanh_value": True` so `load_az_network()`
rebuilds the right class. Checkpoints without the flag (every pre-existing
`checkpoints_az/*.pt` and `checkpoints/ppo_*.pt`) load as plain SplendorNetwork,
so old models keep behaving exactly as they did.
"""
from __future__ import annotations

import torch

from splendor_duel.agents.ppo.network import (
    DEFAULT_HIDDEN_SIZES, SplendorNetwork, infer_hidden_sizes,
)


class AZNetwork(SplendorNetwork):
    """SplendorNetwork with a tanh-bounded value head.

    Parameter shapes are unchanged, so `load_state_dict` works in both
    directions against a plain SplendorNetwork (e.g. `--init` from PPO).
    """

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        logits = self.policy_head(features).clamp(-15, 15)
        value = torch.tanh(self.value_head(features))
        return logits, value


def build_az_network(
    hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
    init_checkpoint: str | None = None,
    device: str = "cpu",
) -> AZNetwork:
    """Fresh AZNetwork, optionally warm-started from any checkpoint.

    When `init_checkpoint` is given its architecture wins — you cannot load
    mismatched shapes, so `hidden_sizes` is only used for a fresh network.
    """
    if init_checkpoint is None:
        return AZNetwork(hidden_sizes=hidden_sizes).to(device)

    data = torch.load(init_checkpoint, map_location=device, weights_only=False)
    state_dict = data["network"] if isinstance(data, dict) and "network" in data else data
    net = AZNetwork(hidden_sizes=infer_hidden_sizes(state_dict)).to(device)
    net.load_state_dict(state_dict)
    return net


def load_az_network(path: str, device: str = "cpu") -> tuple[SplendorNetwork, dict]:
    """Load a checkpoint into the class it was trained as.

    Returns (network, raw_checkpoint_dict). Use this instead of
    `ppo.network.load_network_from_checkpoint` anywhere an AZ checkpoint may be
    involved — the latter always builds an untanh'd SplendorNetwork, which
    would misread the value head of anything trained by train_az_v2.
    """
    data = torch.load(path, map_location=device, weights_only=False)
    state_dict = data["network"] if isinstance(data, dict) and "network" in data else data
    meta = data if isinstance(data, dict) else {}

    cls = AZNetwork if meta.get("tanh_value", False) else SplendorNetwork
    net = cls(hidden_sizes=infer_hidden_sizes(state_dict)).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net, meta
