"""
greedy_by_claude.py — Alternative heuristic agent with 1-ply lookahead (v2).

Changes vs v1 (which lost to Greedy 32/68, GreedyByChatGPT 27/73,
GeminiGreedyAgent 20/80):

    - Progress term weight tripled (1000 → 3000) so victory-path
      progress dominates earlier in the game.
    - Resource accumulation weights cut (bonuses 2.0→1.2,
      tokens 0.5→0.2, reserves 1.5→0.5). v1 was "hoarding-biased":
      avg game length 75 vs 59-68 for other greedies.
    - Added `_affordable_soon_value`: cards in pyramid/reserve weighted
      by 1/(remaining_cost+1). Implicitly targets reachable cards.
    - Token value now depends on whether tokens close gaps to those
      affordable cards, not just their raw colour.
    - Hoarding penalty triggers at >2 of one colour (was >3).
    - Scrolls lookup also checks GameState (not just PlayerState).
"""
from __future__ import annotations

import random

import numpy as np

from splendor_duel.game.actions import Action
from splendor_duel.game.card import Card
from splendor_duel.game.constants import Gem, N_GEMS
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.player import PlayerState
from splendor_duel.game.state import GameState

from .base_agent import BaseAgent

VICTORY_PRESTIGE = 20
VICTORY_CROWNS = 10
VICTORY_MONO = 10

WIN_SCORE = 1_000_000.0
LOSS_SCORE = -1_000_000.0


class GreedyByClaudeV2(BaseAgent):
    """1-ply lookahead greedy with tempo-oriented utility function."""

    def __init__(
            self,
            seed: int | None = None,
            name: str = "GreedyClaude",
    ) -> None:
        super().__init__(name=name)
        self._rng = random.Random(seed)
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
                scores.append(float("-inf"))

        max_score = max(scores)
        eps = 1e-6
        best_indices = [
            i for i, s in enumerate(scores) if s >= max_score - eps
        ]
        return legal_actions[self._rng.choice(best_indices)]

    # ── State evaluation ───────────────────────────────────────────────

    def _evaluate(self, state: GameState) -> float:
        me = state.players[self._me]
        opp = state.players[1 - self._me]

        if state.is_game_over:
            if state.winner == self._me:
                return WIN_SCORE
            if state.winner == (1 - self._me):
                return LOSS_SCORE
            return 0.0

        # 1. Victory-path progress (dominant term, tempo-focused)
        my_progress = self._progress(me)
        opp_progress = self._progress(opp)
        progress_term = (
                3000.0 * (my_progress ** 2)
                - 2400.0 * (opp_progress ** 2)
        )

        # 2. Cards reachable soon — implicit "target card" concept
        my_affordable_cards = self._affordable_cards(state, me)
        opp_affordable_cards = self._affordable_cards(state, opp)
        affordable_me = self._affordable_soon_value(my_affordable_cards)
        affordable_opp = self._affordable_soon_value(opp_affordable_cards)

        # 3. Tokens, targeted at what we can afford
        my_tokens = self._token_value(me, my_affordable_cards)
        opp_tokens = self._token_value(opp, opp_affordable_cards)

        # 4. Resources (muted weights vs v1)
        my_bonuses = float(me.bonuses.sum())
        opp_bonuses = float(opp.bonuses.sum())

        my_reserve = len(me.reserved)
        opp_reserve = len(opp.reserved)

        my_scrolls = self._scrolls(state, me, self._me)
        opp_scrolls = self._scrolls(state, opp, 1 - self._me)

        score = (
                progress_term
                + 1.2 * my_bonuses - 0.9 * opp_bonuses
                + 0.2 * (my_tokens - opp_tokens)
                + 0.5 * my_reserve - 0.3 * opp_reserve
                + 1.5 * my_scrolls - 1.5 * opp_scrolls
                + 2.0 * affordable_me - 1.5 * affordable_opp
                # Raw-stat tails
                + 0.3 * me.points - 0.2 * opp.points
                + 1.0 * me.crowns - 0.8 * opp.crowns
        )

        return score

    # ── Progress helpers ───────────────────────────────────────────────

    def _progress(self, player: PlayerState) -> float:
        """Sorted top-path weighted aggregation (dominant-path strategy)."""
        p_prestige = player.points / VICTORY_PRESTIGE
        p_crowns = player.crowns / VICTORY_CROWNS
        p_mono = self._max_mono_points(player) / VICTORY_MONO

        ps = sorted([p_prestige, p_crowns, p_mono], reverse=True)
        return ps[0] + 0.15 * ps[1] + 0.05 * ps[2]

    def _max_mono_points(self, player: PlayerState) -> float:
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

    # ── Affordable cards (target-card surrogate) ───────────────────────

    def _affordable_cards(
            self, state: GameState, player: PlayerState
    ) -> list[tuple[Card, int]]:
        """(card, remaining_cost) for pyramid + own reserves."""
        out: list[tuple[Card, int]] = []
        pyramid = getattr(state, "pyramid", {})
        for lvl in (1, 2, 3):
            for card in pyramid.get(lvl, []):
                if card is None:
                    continue
                out.append((card, self._remaining_cost(card, player)))
        for card in player.reserved:
            out.append((card, self._remaining_cost(card, player)))
        return out

    def _remaining_cost(self, card: Card, player: PlayerState) -> int:
        """Tokens still needed after bonuses and current tokens."""
        cost = np.array(card.cost, dtype=np.int32)
        after_bonus = np.maximum(cost - player.bonuses, 0)
        # Tokens array may be longer than cost (gold at end) — slice
        own_tokens = player.tokens[: len(after_bonus)]
        remaining = np.maximum(after_bonus - own_tokens, 0)
        return int(remaining.sum())

    def _affordable_soon_value(
            self, affordable_cards: list[tuple[Card, int]]
    ) -> float:
        """Sum card_value / (remaining_cost + 1) across reachable cards."""
        total = 0.0
        for card, remaining in affordable_cards:
            total += self._card_value(card) / (remaining + 1)
        return total

    @staticmethod
    def _card_value(card: Card) -> float:
        """Intrinsic (context-free) value of a card."""
        value = card.points * 3.0 + card.crowns * 2.0
        if getattr(card, "gem_bonus", None) is not None:
            value += float(np.asarray(card.gem_bonus).sum()) * 0.8
        elif getattr(card, "is_wildcard", False):
            value += 0.6
        if getattr(card, "ability", None):
            value += 0.8
        return value

    # ── Tokens ─────────────────────────────────────────────────────────

    def _token_value(
            self,
            player: PlayerState,
            affordable_cards: list[tuple[Card, int]],
    ) -> float:
        """Value tokens higher if they close gaps to reachable cards."""
        tokens = player.tokens

        # Aggregate gap across top-5 closest affordable cards
        if affordable_cards:
            top = sorted(affordable_cards, key=lambda x: x[1])[:5]
            agg_gap = np.zeros(N_GEMS, dtype=np.int32)
            for card, _remaining in top:
                cost = np.array(card.cost, dtype=np.int32)
                need = np.maximum(cost - player.bonuses, 0)
                own_tokens = player.tokens[: len(need)]
                gap = np.maximum(need - own_tokens, 0)
                agg_gap = np.maximum(agg_gap, gap)
        else:
            agg_gap = None

        value = 0.0
        for i in range(len(tokens)):
            t = int(tokens[i])
            if t == 0:
                continue
            if i == Gem.GOLD:
                value += t * 2.2  # joker, always useful
            elif i == Gem.PEARL:
                base = 2.0 if (agg_gap is not None and i < len(agg_gap)
                               and agg_gap[i] > 0) else 1.3
                value += t * base
            else:
                # Targeted colour = 2× value vs untargeted
                base = 1.6 if (agg_gap is not None and i < len(agg_gap)
                               and agg_gap[i] > 0) else 0.8
                value += t * base
            # Sharper hoarding penalty (v1: >3, now >2)
            if t > 2:
                value -= (t - 2) * 0.6
        return value

    # ── Scrolls ────────────────────────────────────────────────────────

    def _scrolls(
            self, state: GameState, player: PlayerState, player_index: int
    ) -> int:
        """
        Number of privilege scrolls held by this player.
        Tries PlayerState attrs, then GameState-level arrays.
        Returns 0 if nothing matches (component silently disabled).
        """
        for attr in ("scrolls", "privileges", "privilege_scrolls",
                     "n_scrolls", "scroll_count"):
            if hasattr(player, attr):
                v = getattr(player, attr)
                try:
                    return int(v() if callable(v) else v)
                except (TypeError, ValueError):
                    pass
        for attr in ("privileges", "scrolls", "privilege_scrolls"):
            if hasattr(state, attr):
                v = getattr(state, attr)
                try:
                    return int(v[player_index])
                except (TypeError, IndexError, KeyError):
                    pass
        return 0
