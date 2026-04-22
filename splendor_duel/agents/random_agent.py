"""
random_agent.py — Uniformly random legal move.

Baseline floor: any competent agent must beat this consistently.
"""
from __future__ import annotations

import random

from splendor_duel.game.actions import Action
from splendor_duel.game.state import GameState
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(name="Random")
        self._rng = random.Random(seed)

    def choose_action(
            self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        return self._rng.choice(legal_actions)
