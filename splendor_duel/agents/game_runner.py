"""
game_runner.py — Play a full game between two agents.

Usage:
    from splendor_duel.agents.game_runner import play_game
    result = play_game(agent0, agent1, "data/cards.json")
    print(result)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState
from .base_agent import BaseAgent


@dataclass
class GameResult:
    """Result of a single game."""
    winner: Optional[int]  # 0, 1, or None for a draw
    victory_type: str  # 'prestige_20', 'crowns_10', 'mono_X'
    turns: int
    steps: int
    scores: tuple[int, int]  # final points
    crowns: tuple[int, int]
    cards: tuple[int, int]  # number of cards bought
    elapsed: float  # wall-clock seconds
    agent_names: tuple[str, str]

    def __repr__(self) -> str:
        outcome = (
            "draw" if self.winner is None
            else f"Player {self.winner} ({self.agent_names[self.winner]}) wins"
        )
        return (
            f"Game: {self.agent_names[0]} vs {self.agent_names[1]} → "
            f"{outcome} "
            f"by {self.victory_type} | "
            f"scores={self.scores} crowns={self.crowns} "
            f"turns={self.turns} ({self.elapsed:.1f}s)"
        )


def play_game(
        agent0: BaseAgent,
        agent1: BaseAgent,
        cards_path: str,
        max_steps: int = 3000,
        verbose: bool = False,
) -> GameResult:
    """
    Play a single game between two agents.

    Args:
        agent0: agent controlling player 0
        agent1: agent controlling player 1
        cards_path: path to cards.json
        max_steps: safety limit to prevent infinite games
        verbose: print each action if True

    Returns:
        GameResult with winner and statistics.
    """
    agents = [agent0, agent1]
    state = GameState.new_game(cards_path)

    agent0.notify_game_start(0)
    agent1.notify_game_start(1)

    t0 = time.time()
    steps = 0

    while not state.is_game_over and steps < max_steps:
        legal = GameEngine.get_legal_actions(state)
        assert legal, (
            f"No legal actions at turn {state.turn}, "
            f"phase {state.phase.name}, player {state.current_player}"
        )

        agent = agents[state.current_player]
        action = agent.choose_action(state, legal)

        assert action in legal, (
            f"{agent.name} returned illegal action {action}"
        )

        if verbose:
            from splendor_duel.viz.replay import describe_action
            desc = describe_action(action, state)
            print(f"  T{state.turn} P{state.current_player} [{state.phase.name}]: {desc}")

        state = GameEngine.apply_action(state, action)
        steps += 1

        if verbose:
            print(f"  T{state.turn} | scores {state.players[0].points}-{state.players[1].points} | crowns {state.players[0].crowns}-{state.players[1].crowns}")

    elapsed = time.time() - t0

    agent0.notify_game_end(state)
    agent1.notify_game_end(state)

    if not state.is_game_over:
        # Safety: game didn't finish in max_steps
        # Declare winner by points
        if state.players[0].points >= state.players[1].points:
            winner = 0
        else:
            winner = 1
        victory_type = "timeout"
    else:
        winner = state.winner
        if winner is None:
            # A finished game with no winner is a draw, not a player-0 win.
            # This used to read `state.winner if ... is not None else 0`, which
            # fabricated a win for player 0 — and run_tournament.py flips seats
            # with `1 - result.winner`, so the fabrication became a player-1
            # win on the return leg.  Silently wrong win-rates, no error.
            victory_type = "draw"
        else:
            victory_type = state.players[winner].check_victory() or "unknown"

    return GameResult(
        winner=winner,
        victory_type=victory_type,
        turns=state.turn,
        steps=steps,
        scores=(state.players[0].points, state.players[1].points),
        crowns=(state.players[0].crowns, state.players[1].crowns),
        cards=(len(state.players[0].cards), len(state.players[1].cards)),
        elapsed=elapsed,
        agent_names=(agent0.name, agent1.name),
    )
