"""
greedy_by_gemini.py — Gemini's Custom Heuristic Agent for Splendor Duel.

Strategy "Triple Threat":
- Evaluates cards strictly based on how much they progress the player towards
  the three specific win conditions: 20 total points, 10 crowns, 10 single-color points.
- Highly values 'extra_turn' and 'take_opponent_gem' abilities.
- Tries to avoid giving the opponent privilege scrolls when taking tokens.
- Discards the tokens that are furthest from being useful.
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
from splendor_duel.game.constants import Gem, N_GEMS
from splendor_duel.game.player import PlayerState
from splendor_duel.game.state import GameState
from splendor_duel.game.board import Board
from .base_agent import BaseAgent


# ── Gemini's Card Valuation ───────────────────────────────────────────────────

def _win_condition_score(card: Card, player: PlayerState) -> float:
    """
    Evaluates a card based on how much % it contributes to the 3 win conditions.
    """
    score = 0.0

    # 1. Total Points Condition (Target: 20)
    # Each point is worth ~5% of the win condition.
    score += (card.points / 20.0) * 100

    # 2. Crowns Condition (Target: 10)
    # Each crown is worth 10% of the win condition.
    score += (card.crowns / 10.0) * 100

    # 3. Single Color Points Condition (Target: 10)
    # We find the player's strongest color and see if this card matches it.
    if card.gem_bonus is not None:
        for i, bonus_val in enumerate(card.gem_bonus):
            if bonus_val > 0:
                current_color_pts = player.points_by_color[i] if hasattr(player, 'points_by_color') else 0
                # If this is already a strong color, points here are extremely valuable
                multiplier = 1.0 + (current_color_pts / 10.0)
                score += (card.points / 10.0) * 100 * multiplier
    elif card.is_wildcard:
        # Wildcards copy the best color, so they are inherently valuable for the 10-point condition
        score += (card.points / 10.0) * 100 * 1.5

    # 4. Abilities Value
    ability_weights = {
        'extra_turn': 30.0,  # Action economy is king
        'take_opponent_gem': 20.0,  # Swingy: +1 for us, -1 for them
        'take_scroll': 10.0,
        'take_same_gem': 10.0,
    }
    if card.ability:
        score += ability_weights.get(card.ability, 0.0)

    # 5. Base economic value (having bonuses makes future cards cheaper)
    if card.gem_bonus is not None or card.is_wildcard:
        score += 5.0

    return score


def _card_efficiency(card: Card, player: PlayerState) -> float:
    """Score adjusted by remaining cost. Divides value by required missing tokens."""
    cost_vec = np.array(card.cost, dtype=np.int8)
    remaining = np.maximum(cost_vec - player.bonuses, 0)
    remaining = np.maximum(remaining - player.tokens[:N_GEMS], 0)
    total_missing = int(remaining.sum())

    # If we can afford it right now, efficiency is just its raw score.
    # Otherwise, penalize it based on how many tokens we still need.
    base_score = _win_condition_score(card, player)
    return base_score / (total_missing + 1.0)


# ── Gemini's Agent Class ──────────────────────────────────────────────────────

class GeminiGreedyAgent(BaseAgent):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(name="GeminiGreedy")
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

    def _get_best_target_card(self, state: GameState) -> Optional[Card]:
        """Finds the most efficient card to work towards."""
        player = state.active
        best_card = None
        best_eff = -1.0

        for lvl in (1, 2, 3):
            for card in state.pyramid.get(lvl, []):
                eff = _card_efficiency(card, player)
                if eff > best_eff:
                    best_eff, best_card = eff, card

        for card in player.reserved:
            eff = _card_efficiency(card, player)
            if eff > best_eff:
                best_eff, best_card = eff, card

        return best_card

    # ── Phase Handlers ────────────────────────────────────────────────────────

    def _choose_optional(self, state: GameState, actions: list[Action]) -> Action:
        # Use scrolls only if we have a target card and the scroll gives us a MISSING token.
        scroll_actions = [a for a in actions if isinstance(a, UseScroll)]
        if scroll_actions:
            target = self._get_best_target_card(state)
            if target:
                cost_vec = np.array(target.cost, dtype=np.int8)
                gap = np.maximum(cost_vec - state.active.bonuses - state.active.tokens, 0)

                best_scroll = None
                max_gap_val = 0
                for a in scroll_actions:
                    gem = state.board.token_at(*a.position)
                    if gem is not None and gap[gem] > max_gap_val:
                        max_gap_val = gap[gem]
                        best_scroll = a

                if best_scroll:
                    return best_scroll

        # Proceed to main; Refilling gives opponent a scroll, avoid if possible.
        proceed = [a for a in actions if isinstance(a, ProceedToMain)]
        if proceed:
            return proceed[0]

        return actions[0]

    def _choose_main(self, state: GameState, actions: list[Action]) -> Action:
        player = state.active

        # 1. Immediate Buy (Greedy core: If we can buy a good card, do it)
        buy_actions = [a for a in actions if isinstance(a, BuyCard)]
        if buy_actions:
            def score_buy(a: BuyCard):
                card = state.pyramid[a.level][a.index] if a.source == 'pyramid' else player.reserved[a.index]
                return _win_condition_score(card, player)

            return max(buy_actions, key=score_buy)

        # 2. Take Tokens towards best target
        take_actions = [a for a in actions if isinstance(a, TakeTokens)]
        target = self._get_best_target_card(state)

        best_take = None
        best_take_score = -9999.0

        if take_actions:
            gap = np.zeros(N_GEMS)
            if target:
                gap = np.maximum(np.array(target.cost, dtype=np.int8) - player.bonuses - player.tokens, 0)

            for action in take_actions:
                taken = np.zeros(N_GEMS, dtype=np.int8)
                for r, c in action.positions:
                    gem = state.board.token_at(r, c)
                    if gem is not None: taken[gem] += 1

                # Base score: how many useful tokens we get
                useful_tokens = np.minimum(taken, gap).sum()
                score = float(useful_tokens) * 10.0

                # Bonus for just getting more tokens (economy)
                score += len(action.positions) * 2.0

                # PENALTY: Avoid giving opponent a scroll (3 same or 2 pearls)
                if Board.triggers_privilege(action.positions, taken):
                    score -= 15.0  # High penalty

                if score > best_take_score:
                    best_take_score = score
                    best_take = action

        # 3. Reserve High Value Card
        reserve_actions = [a for a in actions if isinstance(a, ReserveCard)]
        best_res = None
        best_res_score = -9999.0

        if reserve_actions:
            for action in reserve_actions:
                if action.source == 'deck':
                    score = action.level * 5.0
                else:
                    card = state.pyramid[action.level][action.index]
                    # Reserve is good if card is extremely valuable, to lock it from opponent
                    score = _card_efficiency(card, player) * 0.8

                if score > best_res_score:
                    best_res_score = score
                    best_res = action

        # Choose between taking tokens and reserving
        if best_take and best_take_score >= best_res_score:
            return best_take
        if best_res and best_res_score > best_take_score:
            return best_res
        if best_take:
            return best_take

        return self._rng.choice(actions)

    def _choose_effect(self, state: GameState, actions: list[Action]) -> Action:
        opp_gem = [a for a in actions if isinstance(a, EffectTakeOpponentGem)]
        if opp_gem:
            # Steal the gem the opponent has the most of (cripple their economy)
            return max(opp_gem, key=lambda a: int(state.opponent.tokens[a.gem]))

        wc = [a for a in actions if isinstance(a, EffectChooseWildcard)]
        if wc:
            # Pick wildcard color based on our strongest color path to 10 points
            if hasattr(state.active, 'points_by_color'):
                return max(wc, key=lambda a: state.active.points_by_color.get(a.target_card_index, 0))
            return wc[0]

        return self._rng.choice(actions)

    def _choose_royal(self, state: GameState, actions: list[Action]) -> Action:
        def royal_score(a: ChooseRoyal) -> float:
            royal = state.royal_cards[a.index]
            score = royal.points * 20.0
            if royal.ability == 'extra_turn': score += 50.0
            if royal.ability == 'take_opponent_gem': score += 30.0
            return score

        return max(actions, key=royal_score)

    def _choose_discard(self, state: GameState, actions: list[Action]) -> Action:
        # Discard the token we have the most surplus of.
        # NEVER discard Gold if possible.
        player = state.active

        def discard_penalty(a: DiscardToken) -> float:
            if a.gem == Gem.GOLD: return 1000.0
            target = self._get_best_target_card(state)
            if target:
                gap = np.maximum(np.array(target.cost, dtype=np.int8) - player.bonuses - player.tokens, 0)
                if gap[a.gem] > 0:
                    return 500.0  # We need this for our target card!
            # If not needed, discard the one we have the most of
            return -float(player.tokens[a.gem])

        return min(actions, key=discard_penalty)
