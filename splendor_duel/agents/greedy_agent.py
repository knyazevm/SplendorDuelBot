"""
greedy_agent.py — Heuristic agent for Splendor Duel.

Strategy:
- Buy the most valuable affordable card (points > crowns > ability).
- When taking tokens, prefer colours that reduce the gap to the best
  card in the pyramid or reserve.
- Reserve high-value cards when buying is not possible.
- Scrolls: grab a token that helps the most.
- Discard: drop the least useful colour.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from splendor_duel.game.actions import (
    Action, BuyCard, ChooseRoyal, DiscardToken,
    EffectChooseWildcard, EffectSkip, EffectTakeOpponentGem,
    EffectTakeSameGem, Phase, ProceedToMain, RefillBoard,
    ReserveCard, TakeTokens, UseScroll,
)
from splendor_duel.game.card import Card
from splendor_duel.game.constants import Gem, GEM_NAMES, N_GEMS
from splendor_duel.game.player import PlayerState
from splendor_duel.game.state import GameState
from .base_agent import BaseAgent

# ── Card valuation ────────────────────────────────────────────────────────────

ABILITY_VALUE = {
    'extra_turn': 3.0,
    'take_same_gem': 1.5,
    'take_scroll': 1.0,
    'take_opponent_gem': 2.0,
}


def _card_score(card: Card) -> float:
    """
    Score a card for greedy purchase priority.

    Weights:
    - Points are king (direct path to 20-point victory).
    - Crowns matter for royal unlocks (3 crowns = 2-3 bonus pts).
    - Abilities are a tiebreaker.
    - Bonus gems have hidden value (future cost reduction) but are
      hard to evaluate without lookahead, so we add a small constant.
    """
    score = card.points * 4.0
    score += card.crowns * 2.5
    if card.ability:
        score += ABILITY_VALUE.get(card.ability, 0.0)
    # Having a bonus gem is always worth something
    if card.gem_bonus is not None or card.is_wildcard:
        score += 0.5
    return score


def _card_efficiency(card: Card, player: PlayerState) -> float:
    """
    Score adjusted by how easy the card is to buy.

    efficiency = card_score / (remaining_cost + 1)
    Higher = better bang for the buck.
    """
    cost_vec = np.array(card.cost, dtype=np.int8)
    remaining = np.maximum(cost_vec - player.bonuses, 0)
    remaining = np.maximum(remaining - player.tokens[:N_GEMS], 0)
    total_remaining = int(remaining.sum())
    return _card_score(card) / (total_remaining + 1)


# ── Token gap analysis ────────────────────────────────────────────────────────

def _best_target_card(state: GameState) -> Optional[Card]:
    """
    Find the card with the best efficiency that we might buy soon.
    Considers both pyramid and reserved cards.
    """
    player = state.active
    best_card: Optional[Card] = None
    best_eff = -1.0

    for lvl in (1, 2, 3):
        for card in state.pyramid.get(lvl, []):
            eff = _card_efficiency(card, player)
            if eff > best_eff:
                best_eff = eff
                best_card = card

    for card in player.reserved:
        eff = _card_efficiency(card, player)
        if eff > best_eff:
            best_eff = eff
            best_card = card

    return best_card


def _token_gap(player: PlayerState, card: Card) -> np.ndarray:
    """
    How many more tokens of each colour do we need to buy this card?
    Returns a vector of length N_GEMS (negative = surplus, clamped to 0).
    """
    cost_vec = np.array(card.cost, dtype=np.int8)
    gap = np.maximum(cost_vec - player.bonuses - player.tokens, 0)
    return gap


def _take_tokens_score(
        positions: tuple[tuple[int, int], ...],
        state: GameState,
        target_card: Optional[Card],
) -> float:
    """
    Score a TakeTokens action based on how well the taken gems
    close the gap to the target card.
    """
    board = state.board
    player = state.active

    # Count what we'd take
    taken = np.zeros(N_GEMS, dtype=np.int8)
    for r, c in positions:
        gem = board.token_at(r, c)
        if gem is not None:
            taken[gem] += 1

    if target_card is None:
        # No clear target: prefer taking more tokens, prefer diversity
        n_colours = int(np.count_nonzero(taken[:Gem.PEARL]))  # exclude pearl/gold
        return len(positions) * 1.0 + n_colours * 0.5

    gap = _token_gap(player, target_card)

    # Score = how much of the gap this take fills
    useful = np.minimum(taken, gap)
    score = float(useful.sum()) * 3.0

    # Penalty for taking pearls we don't need
    pearl_needed = gap[Gem.PEARL]
    pearl_taken = taken[Gem.PEARL]
    if pearl_taken > pearl_needed:
        score -= (pearl_taken - pearl_needed) * 1.0

    # Small bonus for taking more tokens overall
    score += len(positions) * 0.3

    # Penalty for triggering opponent privilege (3 same or 2 pearls)
    from splendor_duel.game.board import Board
    if Board.triggers_privilege(positions, taken):
        score -= 2.0

    return score


# ── Greedy agent ──────────────────────────────────────────────────────────────

class GreedyAgent(BaseAgent):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(name="Greedy")
        self._rng = random.Random(seed)

    def choose_action(
            self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        phase = state.phase

        if phase == Phase.OPTIONAL:
            return self._choose_optional(state, legal_actions)
        if phase == Phase.MAIN:
            return self._choose_main(state, legal_actions)
        if phase == Phase.EFFECT:
            return self._choose_effect(state, legal_actions)
        if phase == Phase.ROYAL:
            return self._choose_royal(state, legal_actions)
        if phase == Phase.DISCARD:
            return self._choose_discard(state, legal_actions)

        return self._rng.choice(legal_actions)

    # ── OPTIONAL phase ────────────────────────────────────────────────────

    def _choose_optional(
            self, state: GameState, actions: list[Action]
    ) -> Action:
        # If we have scrolls and there's a useful token, use scroll
        scroll_actions = [a for a in actions if isinstance(a, UseScroll)]
        if scroll_actions:
            target = _best_target_card(state)
            if target is not None:
                gap = _token_gap(state.active, target)
                best_scroll = None
                best_val = -1.0
                for a in scroll_actions:
                    gem = state.board.token_at(*a.position)
                    if gem is not None and gap[gem] > 0:
                        val = float(gap[gem])
                        if val > best_val:
                            best_val = val
                            best_scroll = a
                if best_scroll is not None:
                    return best_scroll

        # Otherwise proceed to main (prefer not to refill — gives opponent a scroll)
        proceed = [a for a in actions if isinstance(a, ProceedToMain)]
        if proceed:
            return proceed[0]

        # Must refill if no main actions possible
        refill = [a for a in actions if isinstance(a, RefillBoard)]
        if refill:
            return refill[0]

        return actions[0]

    # ── MAIN phase ────────────────────────────────────────────────────────

    def _choose_main(
            self, state: GameState, actions: list[Action]
    ) -> Action:
        player = state.active

        # 1. Can we buy? Pick the best card.
        buy_actions = [a for a in actions if isinstance(a, BuyCard)]
        if buy_actions:
            best_buy = max(buy_actions, key=lambda a: self._buy_score(state, a))
            best_score = self._buy_score(state, best_buy)
            if best_score > 0:
                return best_buy

        # 2. Take tokens toward best target card
        take_actions = [a for a in actions if isinstance(a, TakeTokens)]
        reserve_actions = [a for a in actions if isinstance(a, ReserveCard)]
        target = _best_target_card(state)

        if take_actions:
            best_take = max(
                take_actions,
                key=lambda a: _take_tokens_score(a.positions, state, target),
            )
            take_score = _take_tokens_score(
                best_take.positions, state, target
            )
        else:
            best_take = None
            take_score = -999

        # 3. Consider reserving a high-value card
        if reserve_actions:
            best_reserve = max(
                reserve_actions,
                key=lambda a: self._reserve_score(state, a),
            )
            res_score = self._reserve_score(state, best_reserve)
        else:
            best_reserve = None
            res_score = -999

        # Pick the better option
        if best_take and take_score >= res_score:
            return best_take
        if best_reserve and res_score > take_score:
            return best_reserve
        if best_take:
            return best_take

        # Fallback
        return self._rng.choice(actions)

    def _buy_score(self, state: GameState, action: BuyCard) -> float:
        """Score a buy action."""
        if action.source == 'pyramid':
            card = state.pyramid[action.level][action.index]
        else:
            card = state.active.reserved[action.index]
        return _card_score(card)

    def _reserve_score(self, state: GameState, action: ReserveCard) -> float:
        """
        Score a reserve action.

        Reserve is worth it when:
        - The card is high value and we're close to affording it
        - We deny a good card from the opponent
        - We need gold tokens
        """
        if action.source == 'deck':
            # Blind reserve: low score since we don't know the card
            return 1.0 + action.level * 0.5

        card = state.pyramid[action.level][action.index]
        eff = _card_efficiency(card, state.active)
        # Reserve is less valuable than immediate buy, apply discount
        return eff * 0.6

    # ── EFFECT phase ──────────────────────────────────────────────────────

    def _choose_effect(
            self, state: GameState, actions: list[Action]
    ) -> Action:
        # take_same_gem: grab the matching token
        same_gem = [a for a in actions if isinstance(a, EffectTakeSameGem)]
        if same_gem:
            return self._rng.choice(same_gem)  # all equivalent

        # take_opponent_gem: steal the most abundant non-gold gem
        opp_gem = [a for a in actions if isinstance(a, EffectTakeOpponentGem)]
        if opp_gem:
            opponent = state.opponent
            return max(opp_gem, key=lambda a: int(opponent.tokens[a.gem]))

        # choose_wildcard: pick the colour we need most
        wc = [a for a in actions if isinstance(a, EffectChooseWildcard)]
        if wc:
            target = _best_target_card(state)
            if target is not None:
                gap = _token_gap(state.active, target)
                return max(wc, key=lambda a: self._wildcard_value(state, a, gap))
            return wc[0]

        # skip
        return actions[0]

    def _wildcard_value(
            self, state: GameState, action: EffectChooseWildcard, gap: np.ndarray
    ) -> float:
        """Value of placing wildcard on a target card (copying its bonus colour)."""
        target_card = state.active.cards[action.target_card_index]
        if target_card.gem_bonus is not None:
            for i, v in enumerate(target_card.gem_bonus):
                if v > 0:
                    # Prefer colours where we have the biggest gap
                    return float(gap[i]) * v
        return 0.0

    # ── ROYAL phase ───────────────────────────────────────────────────────

    def _choose_royal(
            self, state: GameState, actions: list[Action]
    ) -> Action:
        def royal_score(a: ChooseRoyal) -> float:
            royal = state.royal_cards[a.index]
            score = royal.points * 3.0
            if royal.ability:
                score += ABILITY_VALUE.get(royal.ability, 0.0)
            return score

        return max(actions, key=royal_score)

    # ── DISCARD phase ─────────────────────────────────────────────────────

    def _choose_discard(
            self, state: GameState, actions: list[Action]
    ) -> Action:
        """Discard the least useful token colour."""
        player = state.active
        target = _best_target_card(state)

        if target is not None:
            gap = _token_gap(player, target)

            def discard_cost(a: DiscardToken) -> float:
                # Lower cost = better to discard
                gem = a.gem
                if gem == Gem.GOLD:
                    return 10.0  # never discard gold if possible
                if gap[gem] > 0:
                    return 5.0  # we need this colour
                # Surplus: safe to discard; prefer discarding most abundant
                return -float(player.tokens[gem])

            return min(actions, key=discard_cost)

        # No target: discard most abundant non-gold
        def abundance(a: DiscardToken) -> float:
            if a.gem == Gem.GOLD:
                return -100  # keep gold
            return float(player.tokens[a.gem])

        return max(actions, key=abundance)
