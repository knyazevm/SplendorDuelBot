"""
greedy_by_gemini_v2.py — Gemini's V2 Heuristic Agent for Splendor Duel.

Strategy "Triple Threat & Active Denial":
- Evaluates cards based on 3 win conditions.
- NEW: Actively monitors the opponent's board. If the opponent is close
  to a win condition, drastically increases the value of reserving the cards
  they need (Denial).
- NEW: Rebalanced privilege scroll penalties (it's often worth giving a scroll
  if we get 3 perfect tokens).
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
    """Evaluates a card based on how much % it contributes to the 3 win conditions."""
    score = 0.0

    # 1. Total Points Condition (Target: 20)
    score += (card.points / 20.0) * 100

    # 2. Crowns Condition (Target: 10)
    score += (card.crowns / 10.0) * 100

    # 3. Single Color Points Condition (Target: 10)
    if card.gem_bonus is not None:
        for i, bonus_val in enumerate(card.gem_bonus):
            if bonus_val > 0:
                # Calculate current points in this specific color
                current_color_pts = sum(c.points for c in player.cards if c.gem_bonus and c.gem_bonus[i] > 0)
                multiplier = 1.0 + (current_color_pts / 10.0)
                score += (card.points / 10.0) * 100 * multiplier
    elif card.is_wildcard:
        score += (card.points / 10.0) * 100 * 1.5

    # 4. Abilities Value (Slightly rebalanced for V2)
    ability_weights = {
        'extra_turn': 35.0,
        'take_opponent_gem': 15.0,
        'take_scroll': 10.0,
        'take_same_gem': 10.0,
    }
    if card.ability:
        score += ability_weights.get(card.ability, 0.0)

    if card.gem_bonus is not None or card.is_wildcard:
        score += 5.0

    return score


def _card_efficiency(card: Card, player: PlayerState) -> float:
    """Score adjusted by remaining cost."""
    cost_vec = np.array(card.cost, dtype=np.int8)
    remaining = np.maximum(cost_vec - player.bonuses, 0)
    remaining = np.maximum(remaining - player.tokens[:N_GEMS], 0)
    total_missing = int(remaining.sum())

    base_score = _win_condition_score(card, player)
    return base_score / (total_missing + 1.0)


def _opponent_threat_score(card: Card, opponent: PlayerState) -> float:
    """
    NEW V2: Calculates how dangerous leaving this card on the board is.
    If the opponent is close to winning, and this card pushes them over,
    its threat score skyrockets.
    """
    threat = 0.0
    # Is opponent close to 20 points?
    if opponent.points >= 14:
        if opponent.points + card.points >= 20:
            threat += 500.0  # Critical threat!
        else:
            threat += card.points * 10.0

    # Is opponent close to 10 crowns?
    if opponent.crowns >= 7:
        if opponent.crowns + card.crowns >= 10:
            threat += 500.0
        else:
            threat += card.crowns * 20.0

    # Check mono-color threat
    if opponent.points >= 5 and card.gem_bonus is not None and card.points > 0:
        for i, bonus_val in enumerate(card.gem_bonus):
            if bonus_val > 0:
                opp_color_pts = sum(c.points for c in opponent.cards if c.gem_bonus and c.gem_bonus[i] > 0)
                if opp_color_pts >= 6 and opp_color_pts + card.points >= 10:
                    threat += 500.0
    return threat


# ── Gemini's Agent Class ──────────────────────────────────────────────────────

class GeminiGreedyAgentV2(BaseAgent):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(name="GeminiGreedyV2")
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

        proceed = [a for a in actions if isinstance(a, ProceedToMain)]
        if proceed:
            return proceed[0]

        return actions[0]

    def _choose_main(self, state: GameState, actions: list[Action]) -> Action:
        player = state.active
        opponent = state.opponent

        # 1. Buy Phase
        buy_actions = [a for a in actions if isinstance(a, BuyCard)]
        if buy_actions:
            def score_buy(a: BuyCard):
                card = state.pyramid[a.level][a.index] if a.source == 'pyramid' else player.reserved[a.index]
                return _win_condition_score(card, player)

            return max(buy_actions, key=score_buy)

        # 2. Take Tokens
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

                useful_tokens = np.minimum(taken, gap).sum()
                score = float(useful_tokens) * 10.0
                score += len(action.positions) * 2.0

                # V2: Reduced penalty. Giving a scroll is fine if the tokens are perfectly what we need.
                if Board.triggers_privilege(action.positions, taken):
                    score -= 5.0

                if score > best_take_score:
                    best_take_score = score
                    best_take = action

        # 3. Reserve Phase (V2: Active Denial)
        reserve_actions = [a for a in actions if isinstance(a, ReserveCard)]
        best_res = None
        best_res_score = -9999.0

        if reserve_actions:
            for action in reserve_actions:
                if action.source == 'deck':
                    score = action.level * 5.0
                else:
                    card = state.pyramid[action.level][action.index]
                    my_eff = _card_efficiency(card, player)
                    # V2: Add opponent threat to reserve score.
                    # If they need it, reserving it becomes exponentially more valuable!
                    opp_threat = _opponent_threat_score(card, opponent)
                    score = (my_eff * 0.8) + opp_threat

                if score > best_res_score:
                    best_res_score = score
                    best_res = action

        # Decision
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
            return max(opp_gem, key=lambda a: int(state.opponent.tokens[a.gem]))

        wc = [a for a in actions if isinstance(a, EffectChooseWildcard)]
        if wc:
            player = state.active

            def wc_score(a: EffectChooseWildcard):
                # Pick color we have the most points in to rush mono-color victory
                return sum(c.points for c in player.cards
                           if c.gem_bonus and c.gem_bonus[a.colour] > 0)

            return max(wc, key=wc_score)

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
        player = state.active

        def discard_penalty(a: DiscardToken) -> float:
            if a.gem == Gem.GOLD: return 1000.0
            target = self._get_best_target_card(state)
            if target:
                gap = np.maximum(np.array(target.cost, dtype=np.int8) - player.bonuses - player.tokens, 0)
                if gap[a.gem] > 0:
                    return 500.0
            return -float(player.tokens[a.gem])

        return min(actions, key=discard_penalty)