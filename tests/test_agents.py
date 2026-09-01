"""
Tests for agents: random, greedy, MCTS, and game_runner.
Run with:  pytest tests/test_agents.py -v
"""
import json
import random

import numpy as np
import pytest

from splendor_duel.game.actions import (
    Action, BuyCard, Phase, ProceedToMain, TakeTokens,
)
from splendor_duel.game.board import Board
from splendor_duel.game.card import Card, RoyalCard
from splendor_duel.game.constants import (
    ABILITY_EXTRA_TURN, GEM_NAMES, Gem, N_GEMS,
)
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.player import PlayerState
from splendor_duel.game.state import GameState

from splendor_duel.agents import (
    RandomAgent, GreedyAgent, MCTSAgent, play_game,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

CARDS_JSON = {
    "cards": [
        {
            "id": f"L{lvl}_{i:02d}", "level": lvl,
            "cost": {name: (1 if name == 'white' and lvl == 1
                            else 2 if name == 'white' and lvl == 2
            else 4 if name == 'white' and lvl == 3
            else 0)
                     for name in GEM_NAMES},
            "gem_bonus": {"white": 1} if i % 3 != 0 else {"red": 1},
            "points": lvl,
            "crowns": 1 if i % 4 == 0 else 0,
            "ability": None,
        }
        for lvl in (1, 2, 3)
        for i in range(10 if lvl == 1 else 8 if lvl == 2 else 6)
    ],
    "royal_cards": [
        {"id": "R_01", "points": 2, "ability": "take_scroll"},
        {"id": "R_02", "points": 2, "ability": "extra_turn"},
        {"id": "R_03", "points": 2, "ability": None},
        {"id": "R_04", "points": 3, "ability": None},
    ],
}


@pytest.fixture
def cards_path(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(CARDS_JSON))
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
# RandomAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestRandomAgent:

    def test_always_returns_legal_action(self, cards_path):
        random.seed(123)
        agent = RandomAgent(seed=1)
        state = GameState.new_game(cards_path)
        for _ in range(200):
            if state.is_game_over:
                break
            actions = GameEngine.get_legal_actions(state)
            choice = agent.choose_action(state, actions)
            assert choice in actions
            state = GameEngine.apply_action(state, choice)

    def test_deterministic_with_seed(self, cards_path):
        random.seed(99)
        state1 = GameState.new_game(cards_path)
        random.seed(99)
        state2 = GameState.new_game(cards_path)

        a1 = RandomAgent(seed=42)
        a2 = RandomAgent(seed=42)

        for _ in range(50):
            if state1.is_game_over:
                break
            actions1 = GameEngine.get_legal_actions(state1)
            actions2 = GameEngine.get_legal_actions(state2)
            c1 = a1.choose_action(state1, actions1)
            c2 = a2.choose_action(state2, actions2)
            assert c1 == c2
            # RefillBoard draws from the bag via the GLOBAL random stream, so
            # advancing the two games one after another off that single stream
            # would hand them different boards.  Re-seed symmetrically: this
            # test is about the agent being deterministic, not the engine.
            step_seed = random.randrange(2 ** 31)
            random.seed(step_seed)
            state1 = GameEngine.apply_action(state1, c1)
            random.seed(step_seed)
            state2 = GameEngine.apply_action(state2, c2)
            assert state1.board == state2.board


# ══════════════════════════════════════════════════════════════════════════════
# GreedyAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestGreedyAgent:

    def test_always_returns_legal_action(self, cards_path):
        random.seed(42)
        agent = GreedyAgent(seed=1)
        state = GameState.new_game(cards_path)
        for _ in range(500):
            if state.is_game_over:
                break
            actions = GameEngine.get_legal_actions(state)
            choice = agent.choose_action(state, actions)
            assert choice in actions
            state = GameEngine.apply_action(state, choice)

    def test_prefers_buying_over_taking(self, cards_path):
        """When a card is affordable, greedy should buy it."""
        random.seed(42)
        state = GameState.new_game(cards_path)
        # Give player lots of tokens
        state.players[state.current_player].tokens[Gem.WHITE] = 5
        state.players[state.current_player].tokens[Gem.RED] = 3
        state.phase = Phase.MAIN

        agent = GreedyAgent(seed=1)
        actions = GameEngine.get_legal_actions(state)
        buy_actions = [a for a in actions if isinstance(a, BuyCard)]
        if buy_actions:
            choice = agent.choose_action(state, actions)
            assert isinstance(choice, BuyCard)

    def test_completes_full_game(self, cards_path):
        random.seed(42)
        result = play_game(GreedyAgent(), GreedyAgent(), cards_path)
        assert result.winner in (0, 1)
        assert result.turns > 0
        assert result.victory_type != "timeout"


# ══════════════════════════════════════════════════════════════════════════════
# MCTSAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestMCTSAgent:

    def test_always_returns_legal_action(self, cards_path):
        random.seed(42)
        agent = MCTSAgent(iterations=20, seed=1)
        state = GameState.new_game(cards_path)
        for _ in range(100):
            if state.is_game_over:
                break
            actions = GameEngine.get_legal_actions(state)
            choice = agent.choose_action(state, actions)
            assert choice in actions
            state = GameEngine.apply_action(state, choice)

    def test_single_action_returns_immediately(self, cards_path):
        """With only one legal action, MCTS shouldn't waste iterations."""
        random.seed(42)
        state = GameState.new_game(cards_path)
        agent = MCTSAgent(iterations=1000, seed=1)
        # Find a state with single action
        for _ in range(200):
            if state.is_game_over:
                break
            actions = GameEngine.get_legal_actions(state)
            if len(actions) == 1:
                import time
                t0 = time.time()
                choice = agent.choose_action(state, actions)
                elapsed = time.time() - t0
                assert choice == actions[0]
                assert elapsed < 0.1  # should be instant
                break
            state = GameEngine.apply_action(state, actions[0])

    def test_completes_full_game(self, cards_path):
        random.seed(42)
        result = play_game(
            MCTSAgent(iterations=30, seed=1),
            RandomAgent(seed=2),
            cards_path,
        )
        assert result.winner in (0, 1)
        assert result.victory_type != "timeout"


# ══════════════════════════════════════════════════════════════════════════════
# GameRunner
# ══════════════════════════════════════════════════════════════════════════════

class TestGameRunner:

    def test_play_game_returns_result(self, cards_path):
        result = play_game(RandomAgent(seed=1), RandomAgent(seed=2), cards_path)
        assert result.winner in (0, 1)
        assert result.turns > 0
        assert result.steps > 0
        assert result.elapsed > 0
        assert len(result.agent_names) == 2

    def test_game_terminates(self, cards_path):
        """Play 10 random-vs-random games, all must terminate."""
        for seed in range(10):
            random.seed(seed)
            result = play_game(
                RandomAgent(seed=seed),
                RandomAgent(seed=seed + 100),
                cards_path,
            )
            assert result.victory_type != "timeout", (
                f"Game {seed} timed out at turn {result.turns}"
            )

    def test_greedy_vs_random(self, cards_path):
        """Greedy should win majority against random over 20 games."""
        greedy_wins = 0
        n_games = 20
        for i in range(n_games):
            random.seed(i)
            if i % 2 == 0:
                result = play_game(GreedyAgent(seed=i), RandomAgent(seed=i + 50), cards_path)
                if result.winner == 0:
                    greedy_wins += 1
            else:
                result = play_game(RandomAgent(seed=i + 50), GreedyAgent(seed=i), cards_path)
                if result.winner == 1:
                    greedy_wins += 1
        # Greedy should win at least 60% — if not, heuristic is broken
        assert greedy_wins >= 12, (
            f"Greedy only won {greedy_wins}/{n_games} vs Random"
        )
