"""
base_agent.py — Abstract agent interface.

All agents implement choose_action(state, actions) → action.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from splendor_duel.game.actions import Action
from splendor_duel.game.state import GameState


class BaseAgent(ABC):
    """Abstract base for all Splendor Duel agents."""

    def __init__(self, name: str = "Agent") -> None:
        self.name = name

    @abstractmethod
    def choose_action(
            self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        """
        Pick one action from the list of legal actions.

        Args:
            state: current game state (read-only — agents must NOT mutate it)
            legal_actions: non-empty list of legal actions for this state

        Returns:
            One action from legal_actions.
        """
        ...

    def notify_game_start(self, player_index: int) -> None:
        """Called at the start of a game. Override if agent needs setup."""
        pass

    def notify_game_end(self, state: GameState) -> None:
        """Called when the game ends. Override for learning agents."""
        pass

    def __repr__(self) -> str:
        return self.name
