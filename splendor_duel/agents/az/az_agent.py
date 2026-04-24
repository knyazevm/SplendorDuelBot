"""
az_agent.py — BaseAgent wrapper for trained AlphaZero network.

For play (not training), we disable dirichlet noise, use argmax over
visit counts, and typically use more MCTS simulations than during training.

Usage:
    agent = AZAgent.load("checkpoints_az/az_final.pt", n_simulations=200)
    result = play_game(agent, GreedyAgent(), "data/cards.json")
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from splendor_duel.agents.base_agent import BaseAgent
from splendor_duel.agents.ppo.network import SplendorNetwork
from splendor_duel.env.action_map import action_to_index, index_to_action
from splendor_duel.game.actions import Action
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState

from .mcts_az import NetworkEvaluator, run_mcts


class AZAgent(BaseAgent):
    """
    AlphaZero agent for play.

    Parameters:
        network:        trained SplendorNetwork
        n_simulations:  MCTS simulations per move (more = stronger, slower)
        c_puct:         PUCT constant
        deterministic:  if True, argmax over visits; else sample ∝ visits
        device:         "cpu" or "cuda"
    """

    def __init__(
            self,
            network: SplendorNetwork,
            n_simulations: int = 200,
            c_puct: float = 1.5,
            deterministic: bool = True,
            device: str = "cpu",
            name: Optional[str] = None,
    ):
        super().__init__(name=name or f"AZ({n_simulations})")
        self.network = network
        self.network.eval()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.deterministic = deterministic
        self.device = torch.device(device)
        self._evaluator = NetworkEvaluator(network, device=device)

    @classmethod
    def load(
            cls,
            path: str,
            n_simulations: int = 200,
            c_puct: float = 1.5,
            deterministic: bool = True,
            device: str = "cpu",
    ) -> "AZAgent":
        network = SplendorNetwork()
        data = torch.load(path, map_location=device, weights_only=False)
        if "network" in data:
            network.load_state_dict(data["network"])
        else:
            network.load_state_dict(data)
        network.to(device)
        iters = data.get("total_iterations", "?") if isinstance(data, dict) else "?"
        return cls(
            network=network,
            n_simulations=n_simulations,
            c_puct=c_puct,
            deterministic=deterministic,
            device=device,
            name=f"AZ({n_simulations},it{iters})",
        )

    def choose_action(
            self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        # Run MCTS
        visits, _, resolved = run_mcts(
            state, self._evaluator,
            n_simulations=self.n_simulations,
            c_puct=self.c_puct,
            dirichlet_eps=0.0,  # no exploration noise during play
        )

        # If MCTS had no decisions (trivial phase resolved to game-over or similar)
        if visits.sum() == 0:
            # Fall back to first legal action from resolved state
            actions = GameEngine.get_legal_actions(resolved)
            if actions:
                return self._map_to_original(actions[0], legal_actions)
            return legal_actions[0]

        if self.deterministic:
            action_idx = int(np.argmax(visits))
        else:
            probs = visits / visits.sum()
            action_idx = int(np.random.choice(len(probs), p=probs))

        # Convert index to Action
        try:
            chosen = index_to_action(action_idx)
        except Exception:
            return legal_actions[0]

        # Because MCTS may have auto-resolved phases (OPTIONAL/DISCARD),
        # `resolved` state may differ from `state`. Check if chosen action
        # is in the caller's legal_actions list; if not, fall back.
        for a in legal_actions:
            if action_to_index(a) == action_idx:
                return a

        # Caller's state was pre-resolution phase — but game_runner gives us
        # a fresh legal_actions list matching current state. If mismatch,
        # return first legal action as safe fallback.
        return legal_actions[0]

    def _map_to_original(self, action: Action, legal_actions: list[Action]) -> Action:
        """Try to find matching action in caller's legal_actions list."""
        target_idx = action_to_index(action)
        for a in legal_actions:
            if action_to_index(a) == target_idx:
                return a
        return legal_actions[0]
