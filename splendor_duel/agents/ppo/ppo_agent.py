"""
ppo_agent.py — BaseAgent wrapper for trained PPO model.

Usage:
    agent = PPOAgent.load("checkpoints/ppo_final.pt")
    # Use in tournaments or web play
    result = play_game(agent, GreedyAgent(), "data/cards.json")
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from splendor_duel.game.actions import Action
from splendor_duel.game.state import GameState
from splendor_duel.env.observation import encode_state
from splendor_duel.env.action_map import action_to_index, legal_mask, N_ACTIONS
from ..base_agent import BaseAgent
from .network import SplendorNetwork


class PPOAgent(BaseAgent):
    """
    Play using a trained PPO policy.

    Parameters:
        network:       trained SplendorNetwork
        deterministic: if True, always pick argmax (no sampling)
        device:        "cpu" or "cuda"
    """

    def __init__(
            self,
            network: SplendorNetwork,
            deterministic: bool = True,
            device: str = "cpu",
            name: str = "PPO",
    ):
        super().__init__(name=name)
        self.network = network
        self.network.eval()
        self.deterministic = deterministic
        self.device = torch.device(device)

    @classmethod
    def load(
            cls,
            path: str,
            deterministic: bool = True,
            device: str = "cpu",
    ) -> PPOAgent:
        """Load a trained model from checkpoint."""
        network = SplendorNetwork()
        data = torch.load(path, map_location=device, weights_only=False)
        network.load_state_dict(data["network"])
        network.to(device)
        name = f"PPO({data.get('total_updates', '?')})"
        return cls(network=network, deterministic=deterministic, device=device, name=name)

    def choose_action(
            self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        obs = encode_state(state)
        mask = legal_mask(state)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)

        action_idx, _, _ = self.network.get_action(obs_t, mask_t, self.deterministic)

        # Map back to game Action — find it in legal_actions
        for a in legal_actions:
            if action_to_index(a) == action_idx:
                return a

        # Fallback (shouldn't happen if mask is correct)
        return legal_actions[0]
