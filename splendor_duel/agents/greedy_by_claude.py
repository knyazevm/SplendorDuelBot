"""
greedy_by_claude.py — Alternative heuristic agent with 1-ply lookahead.

Strategy differs from the phase-dispatching GreedyAgent:

    1. For every legal action, simulate the result via
       GameEngine.apply_action and evaluate the resulting state
       with a single unified utility function.
    2. Pick the action with the highest utility (random tie-break).

The state utility reflects the three victory paths from the rulebook
(20 prestige, 10 crowns, 10 mono-colour) plus the opponent's progress
along the same three paths as a defensive term.

Design notes:

    - Quadratic weighting of progress: moving from 0.7 to 0.8 matters
      more than moving from 0.1 to 0.2 (closer to goal = more precious).
    - Dominant-path commitment: agent primarily optimises the best path
      but keeps a small weight on secondary paths (prefers actions that
      improve two axes at once over actions improving only one).
    - Bonuses > tokens > reserves, gold > pearl > coloured token.
    - Soft penalty on hoarding one colour past 3 tokens (10-token cap
      at end of turn).
"""
from __future__ import annotations

import random

import numpy as np

from splendor_duel.game.actions import Action
from splendor_duel.game.constants import Gem, N_GEMS
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.player import PlayerState
from splendor_duel.game.state import GameState

from .base_agent import BaseAgent

# Victory thresholds (from the rulebook)
VICTORY_PRESTIGE = 20
VICTORY_CROWNS = 10
VICTORY_MONO = 10

# Terminal state scores
WIN_SCORE = 1_000_000.0
LOSS_SCORE = -1_000_000.0


class GreedyByClaude(BaseAgent):
    """
    Heuristic agent with 1-ply lookahead.

    Single decision rule across all phases:
        argmax over legal_actions of evaluate(apply_action(state, a)).
    """

    def __init__(
            self,
            seed: int | None = None,
            name: str = "GreedyClaude",
    ) -> None:
        super().__init__(name=name)
        self._rng = random.Random(seed)
        # Our player index — set by notify_game_start
        self._me: int = 0

    def notify_game_start(self, player_index: int) -> None:
        self._me = player_index

    # ── Decision loop ──────────────────────────────────────────────────

    def choose_action(
            self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        scores: list[float] = []
        for action in legal_actions:
            try:
                next_state = GameEngine.apply_action(state, action)
                scores.append(self._evaluate(next_state))
            except Exception:
                # Defensive: if simulation fails, treat as worst option
                scores.append(float("-inf"))

        max_score = max(scores)
        # Random tie-break among actions within epsilon of the best
        eps = 1e-6
        best_indices = [
            i for i, s in enumerate(scores) if s >= max_score - eps
        ]
        return legal_actions[self._rng.choice(best_indices)]

    # ── State evaluation ───────────────────────────────────────────────

    def _evaluate(self, state: GameState) -> float:
        """
        Utility of ``state`` from the perspective of self._me.
        Higher is better for us.
        """
        me = state.players[self._me]
        opp = state.players[1 - self._me]

        # Terminal shortcut
        if state.is_game_over:
            if state.winner == self._me:
                return WIN_SCORE
            if state.winner == (1 - self._me):
                return LOSS_SCORE
            return 0.0

        # Progress along the three victory paths, aggregated
        my_progress = self._progress(me)
        opp_progress = self._progress(opp)

        # Quadratic — closer to goal grows disproportionately
        progress_term = (
                1000.0 * (my_progress ** 2)
                - 800.0 * (opp_progress ** 2)
        )

        # Resource components
        my_bonuses = float(me.bonuses.sum())
        opp_bonuses = float(opp.bonuses.sum())

        my_tokens = self._token_value(me)
        opp_tokens = self._token_value(opp)

        my_reserve = len(me.reserved)
        opp_reserve = len(opp.reserved)

        my_scrolls = self._scrolls(me)
        opp_scrolls = self._scrolls(opp)

        score = (
                progress_term
                # Bonuses = permanent discount; most valuable persistent resource
                + 2.0 * my_bonuses - 1.3 * opp_bonuses
                # Tokens are consumable, but still useful — diluted weight
                + 0.5 * (my_tokens - opp_tokens)
                # Reserves are options; slightly bigger penalty on us if asymmetric
                + 1.5 * my_reserve - 0.8 * opp_reserve
                # Scrolls ~= free token in the future
                + 1.2 * my_scrolls - 1.2 * opp_scrolls
                # Small raw-stat tails for stability when progress saturates
                + 0.5 * me.points - 0.4 * opp.points
                + 1.5 * me.crowns - 1.2 * opp.crowns
        )

        return score

    # ── Helpers ────────────────────────────────────────────────────────

    def _progress(self, player: PlayerState) -> float:
        """
        Weighted aggregation of progress along all three victory paths.

        Weights 1.0 / 0.15 / 0.05 (sorted descending) encode the
        dominant-path strategy: pick one path but value diversification
        on secondary paths slightly.
        """
        p_prestige = player.points / VICTORY_PRESTIGE
        p_crowns = player.crowns / VICTORY_CROWNS
        p_mono = self._max_mono_points(player) / VICTORY_MONO

        ps = sorted([p_prestige, p_crowns, p_mono], reverse=True)
        return ps[0] + 0.15 * ps[1] + 0.05 * ps[2]

    def _max_mono_points(self, player: PlayerState) -> float:
        """
        Max prestige points accumulated in a single bonus-colour column.

        Cards with no points don't contribute. Cards with a known colour
        bonus go to that column. Wildcard cards without an assigned
        colour are added optimistically to the best column (upper bound).
        """
        totals = np.zeros(N_GEMS, dtype=np.int32)
        unassigned = 0
        for card in player.cards:
            if card.points <= 0:
                continue
            assigned_color = None
            bonus = getattr(card, "gem_bonus", None)
            if bonus is not None:
                for i, v in enumerate(bonus):
                    if v > 0:
                        assigned_color = i
                        break
            if assigned_color is not None:
                totals[assigned_color] += card.points
            else:
                unassigned += card.points
        best = int(totals.max()) if totals.size else 0
        return float(best + unassigned)

    def _token_value(self, player: PlayerState) -> float:
        """
        Resource value of tokens with diminishing returns.

        Gold is most valuable (joker on any colour), pearl next
        (scarce — only 2 in the bag), coloured tokens baseline.
        Mild penalty above 3 of one colour to discourage hoarding
        against the 10-token end-of-turn cap.
        """
        value = 0.0
        tokens = player.tokens
        for i in range(len(tokens)):
            t = int(tokens[i])
            if i == Gem.GOLD:
                value += t * 2.0
            elif i == Gem.PEARL:
                value += t * 1.3
            else:
                value += t * 1.0
            if t > 3:
                value -= (t - 3) * 0.4
        return value

    def _scrolls(self, player: PlayerState) -> int:
        """
        Number of privilege scrolls held by the player.
        Attribute name unknown in advance — try common candidates.
        """
        for attr in ("scrolls", "privileges", "privilege_scrolls", "n_scrolls"):
            if hasattr(player, attr):
                v = getattr(player, attr)
                return int(v() if callable(v) else v)
        return 0
