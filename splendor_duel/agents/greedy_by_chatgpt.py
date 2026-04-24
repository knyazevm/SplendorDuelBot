"""
greedy_by_chatgpt.py — stronger heuristic agent for Splendor Duel.

Design goals:
- Prefer immediate wins.
- Prefer buying valuable cards over preparing forever.
- Track crowns and mono-colour prestige as real win paths.
- Take tokens that make strong cards affordable soon.
- Avoid giving opponent privileges unless the move is clearly worth it.
- Reserve cards not only for ourselves, but also to deny opponent threats.
- Handle effects, royal choices, and discard decisions with the same target logic.

This is still NOT a learning model.
It is a hand-written heuristic policy:
    state + legal actions -> scored actions -> best action
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from splendor_duel.game.actions import (
    Action,
    BuyCard,
    ChooseRoyal,
    DiscardToken,
    EffectChooseWildcard,
    EffectSkip,
    EffectTakeOpponentGem,
    EffectTakeSameGem,
    Phase,
    ProceedToMain,
    RefillBoard,
    ReserveCard,
    TakeTokens,
    UseScroll,
)
from splendor_duel.game.card import Card
from splendor_duel.game.constants import Gem, N_GEMS
from splendor_duel.game.player import PlayerState
from splendor_duel.game.state import GameState
from splendor_duel.game.board import Board
from .base_agent import BaseAgent

ABILITY_VALUE = {
    "extra_turn": 4.0,
    "take_opponent_gem": 2.5,
    "take_same_gem": 1.8,
    "take_scroll": 1.4,
}


class GreedyByChatGPT(BaseAgent):
    """
    A stronger greedy baseline.

    Main priorities:
    1. Win immediately if possible.
    2. Block opponent's immediate win if possible.
    3. Buy the best available card.
    4. Take tokens toward a high-value reachable target.
    5. Reserve high-value or dangerous opponent cards.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(name="GreedyByChatGPT")
        self._rng = random.Random(seed)

    def choose_action(
            self,
            state: GameState,
            legal_actions: list[Action],
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

    # ---------------------------------------------------------------------
    # MAIN PHASE
    # ---------------------------------------------------------------------

    def _choose_main(self, state: GameState, actions: list[Action]) -> Action:
        buy_actions = [a for a in actions if isinstance(a, BuyCard)]
        take_actions = [a for a in actions if isinstance(a, TakeTokens)]
        reserve_actions = [a for a in actions if isinstance(a, ReserveCard)]

        # 1. Immediate winning buy.
        if buy_actions:
            winning_buys = [
                a for a in buy_actions
                if self._buy_wins_immediately(state, a)
            ]
            if winning_buys:
                return max(winning_buys, key=lambda a: self._buy_score(state, a))

        # 2. Normal best buy.
        if buy_actions:
            best_buy = max(buy_actions, key=lambda a: self._buy_score(state, a))
            best_buy_score = self._buy_score(state, best_buy)

            # Buying is usually very good in Splendor Duel.
            # But avoid buying nearly worthless cards if another move is much better.
            if best_buy_score >= 3.0:
                return best_buy

        target = self._best_target_card(state)

        # 3. Score best token-taking move.
        best_take: Optional[TakeTokens] = None
        take_score = -10_000.0
        if take_actions:
            best_take = max(
                take_actions,
                key=lambda a: self._take_tokens_score(state, a, target),
            )
            take_score = self._take_tokens_score(state, best_take, target)

        # 4. Score best reserve move.
        best_reserve: Optional[ReserveCard] = None
        reserve_score = -10_000.0
        if reserve_actions:
            best_reserve = max(
                reserve_actions,
                key=lambda a: self._reserve_score(state, a),
            )
            reserve_score = self._reserve_score(state, best_reserve)

        # 5. Choose between taking tokens and reserving.
        if best_reserve is not None and reserve_score > take_score:
            return best_reserve
        if best_take is not None:
            return best_take
        if best_reserve is not None:
            return best_reserve

        return self._rng.choice(actions)

    # ---------------------------------------------------------------------
    # CARD SCORING
    # ---------------------------------------------------------------------

    def _buy_score(self, state: GameState, action: BuyCard) -> float:
        card = self._card_from_buy_action(state, action)
        player = state.active
        opponent = state.opponent

        score = self._card_intrinsic_score(card, player)

        # Immediate victory is huge.
        if self._card_would_win(player, card):
            score += 10_000.0

        # Strongly value cards that move us close to a victory route.
        score += self._victory_progress_bonus(player, card)

        # Slightly value denying opponent's visible plan if this was a public card.
        if action.source == "pyramid":
            if self._card_would_win(opponent, card):
                score += 500.0
            score += self._opponent_denial_value(opponent, card) * 0.25

        return score

    def _card_intrinsic_score(self, card: Card, player: PlayerState) -> float:
        """
        Raw card value.

        Points are the most direct win condition.
        Crowns are strong because they unlock royal cards and can win directly.
        Abilities are tactical tempo.
        Bonus colours matter because they discount future purchases.
        """
        score = 0.0

        score += float(card.points) * 5.0
        score += float(card.crowns) * 3.5

        if card.ability:
            score += ABILITY_VALUE.get(card.ability, 0.0)

        bonus_colour = self._bonus_colour(card)
        if bonus_colour is not None:
            score += 0.9

            # Bonus is more useful if we lack that colour in discounts.
            try:
                score += max(0.0, 2.0 - float(player.bonuses[bonus_colour])) * 0.25
            except Exception:
                pass

        if getattr(card, "is_wildcard", False):
            score += 1.3

        return score

    def _card_efficiency(self, card: Card, player: PlayerState) -> float:
        """
        Value adjusted by distance to affordability.
        """
        missing = self._missing_after_tokens(player, card)
        distance = int(missing.sum())

        gold = self._gold_count(player)
        effective_distance = max(0, distance - gold)

        return self._card_intrinsic_score(card, player) / (effective_distance + 1)

    def _best_target_card(self, state: GameState) -> Optional[Card]:
        """
        Pick a near-term target card.

        Unlike the original greedy agent, this version gives extra attention to:
        - cards that approach immediate victory,
        - cards that approach 10 crowns,
        - cards that approach 10 points in one bonus colour.
        """
        player = state.active

        best_card: Optional[Card] = None
        best_score = -10_000.0

        for card in self._visible_and_reserved_cards(state, player):
            eff = self._card_efficiency(card, player)
            progress = self._victory_progress_bonus(player, card)
            score = eff + progress * 0.35

            if self._card_would_win(player, card):
                score += 10_000.0

            if score > best_score:
                best_score = score
                best_card = card

        return best_card

    # ---------------------------------------------------------------------
    # TOKEN SCORING
    # ---------------------------------------------------------------------

    def _take_tokens_score(
            self,
            state: GameState,
            action: TakeTokens,
            target: Optional[Card],
    ) -> float:
        board = state.board
        player = state.active

        taken = np.zeros(N_GEMS, dtype=np.int16)
        for r, c in action.positions:
            gem = board.token_at(r, c)
            if gem is not None and int(gem) < N_GEMS:
                taken[int(gem)] += 1

        score = 0.0

        # Basic value: more tokens are usually better.
        score += len(action.positions) * 0.45

        # Diversity helps flexibility.
        score += int(np.count_nonzero(taken)) * 0.20

        if target is not None:
            missing = self._missing_after_tokens(player, target)
            useful = np.minimum(taken, missing)

            # Main reward: closing the gap to target card.
            score += float(useful.sum()) * 3.5

            # If this move makes the target affordable, that is very important.
            after_missing = np.maximum(missing - taken, 0)
            if int(after_missing.sum()) <= self._gold_count(player):
                score += 4.0

            # Penalize tokens that do not help target, but only mildly:
            # extra tokens can still help future cards.
            waste = np.maximum(taken - missing, 0)
            score -= float(waste.sum()) * 0.35

            # Pearl is valuable, but extra pearl can trigger privilege cost.
            try:
                pearl = int(Gem.PEARL)
                if taken[pearl] > missing[pearl]:
                    score -= float(taken[pearl] - missing[pearl]) * 0.75
            except Exception:
                pass
        else:
            # No target: prefer flexible resource intake.
            score += int(np.count_nonzero(taken)) * 0.60

        # Avoid giving opponent privilege unless the move is good.
        if Board.triggers_privilege(action.positions, taken):
            score -= 2.2

        # Bonus: if taken tokens help several good near-term cards.
        score += self._multi_card_token_utility(state, taken) * 0.35

        return score

    def _multi_card_token_utility(self, state: GameState, taken: np.ndarray) -> float:
        """
        How useful are these tokens across several plausible future cards?
        """
        player = state.active
        candidates = self._visible_and_reserved_cards(state, player)

        scored = []
        for card in candidates:
            missing = self._missing_after_tokens(player, card)
            useful = float(np.minimum(taken, missing).sum())
            if useful > 0:
                scored.append(useful * self._card_efficiency(card, player))

        if not scored:
            return 0.0

        scored.sort(reverse=True)
        return sum(scored[:3])

    # ---------------------------------------------------------------------
    # RESERVE SCORING
    # ---------------------------------------------------------------------

    def _reserve_score(self, state: GameState, action: ReserveCard) -> float:
        player = state.active
        opponent = state.opponent

        if action.source == "deck":
            # Blind reserve: acceptable mainly for gold and possible high-level upside.
            return 1.0 + float(action.level) * 0.65

        card = state.pyramid[action.level][action.index]

        own_value = self._card_efficiency(card, player) * 0.75
        denial = self._opponent_denial_value(opponent, card)

        score = own_value + denial

        # Reserving a card that would let opponent win is extremely valuable.
        if self._card_would_win(opponent, card):
            score += 1000.0

        # Gold from reserve is useful, especially when close to a target.
        target = self._best_target_card(state)
        if target is not None:
            missing = self._missing_after_tokens(player, target)
            if int(missing.sum()) > 0:
                score += 1.0

        return score

    def _opponent_denial_value(self, opponent: PlayerState, card: Card) -> float:
        """
        How bad would it be if opponent got this card?
        """
        score = 0.0

        score += self._card_intrinsic_score(card, opponent) * 0.25
        score += self._victory_progress_bonus(opponent, card) * 0.45

        missing = self._missing_after_tokens(opponent, card)
        if int(missing.sum()) <= self._gold_count(opponent):
            score += 3.0

        return score

    # ---------------------------------------------------------------------
    # OPTIONAL PHASE
    # ---------------------------------------------------------------------

    def _choose_optional(self, state: GameState, actions: list[Action]) -> Action:
        scroll_actions = [a for a in actions if isinstance(a, UseScroll)]

        if scroll_actions:
            target = self._best_target_card(state)

            if target is not None:
                missing = self._missing_after_tokens(state.active, target)

                best_scroll = None
                best_score = -10_000.0

                for action in scroll_actions:
                    gem = state.board.token_at(*action.position)
                    if gem is None or int(gem) >= N_GEMS:
                        continue

                    score = 0.0

                    if missing[int(gem)] > 0:
                        score += 4.0 + float(missing[int(gem)])

                    # Extra value if this scroll makes the target affordable.
                    after_missing = missing.copy()
                    after_missing[int(gem)] = max(0, after_missing[int(gem)] - 1)
                    if int(after_missing.sum()) <= self._gold_count(state.active):
                        score += 4.0

                    if score > best_score:
                        best_score = score
                        best_scroll = action

                if best_scroll is not None and best_score > 0:
                    return best_scroll

        # Usually avoid refill because opponent receives a privilege.
        proceed = [a for a in actions if isinstance(a, ProceedToMain)]
        if proceed:
            return proceed[0]

        refill = [a for a in actions if isinstance(a, RefillBoard)]
        if refill:
            return refill[0]

        return actions[0]

    # ---------------------------------------------------------------------
    # EFFECT PHASE
    # ---------------------------------------------------------------------

    def _choose_effect(self, state: GameState, actions: list[Action]) -> Action:
        same_gem = [a for a in actions if isinstance(a, EffectTakeSameGem)]
        if same_gem:
            target = self._best_target_card(state)

            if target is not None:
                missing = self._missing_after_tokens(state.active, target)

                def same_gem_score(a: EffectTakeSameGem) -> float:
                    """
                    EffectTakeSameGem in this project appears to identify the token
                    by board position, not by a direct .gem attribute.
                    """
                    position = getattr(a, "position", None)
                    if position is None:
                        return 0.0

                    gem = state.board.token_at(*position)
                    if gem is None:
                        return 0.0

                    gem_i = int(gem)
                    if gem_i >= len(missing):
                        return 0.0

                    return float(missing[gem_i])

                return max(same_gem, key=same_gem_score)

            return self._rng.choice(same_gem)

        opponent_gem = [a for a in actions if isinstance(a, EffectTakeOpponentGem)]
        if opponent_gem:
            opponent = state.opponent

            def steal_score(a: EffectTakeOpponentGem) -> float:
                gem = int(a.gem)
                score = float(opponent.tokens[gem])

                # Prefer stealing colours opponent may need for strong visible cards.
                for card in self._visible_and_reserved_cards_for_player_guess(state, opponent):
                    missing = self._missing_after_tokens(opponent, card)
                    if gem < len(missing) and missing[gem] > 0:
                        score += self._card_efficiency(card, opponent) * 0.15

                return score

            return max(opponent_gem, key=steal_score)

        wildcard = [a for a in actions if isinstance(a, EffectChooseWildcard)]
        if wildcard:
            target = self._best_target_card(state)
            if target is not None:
                missing = self._missing_after_tokens(state.active, target)
                return max(
                    wildcard,
                    key=lambda a: self._wildcard_value(state, a, missing),
                )
            return wildcard[0]

        skip = [a for a in actions if isinstance(a, EffectSkip)]
        if skip:
            return skip[0]

        return actions[0]

    def _wildcard_value(
            self,
            state: GameState,
            action: EffectChooseWildcard,
            missing: np.ndarray,
    ) -> float:
        try:
            target_card = state.active.cards[action.target_card_index]
        except Exception:
            return 0.0

        colour = self._bonus_colour(target_card)
        if colour is None:
            return 0.0

        return float(missing[colour]) + 0.25

    # ---------------------------------------------------------------------
    # ROYAL PHASE
    # ---------------------------------------------------------------------

    def _choose_royal(self, state: GameState, actions: list[Action]) -> Action:
        player = state.active

        def royal_score(action: ChooseRoyal) -> float:
            royal = state.royal_cards[action.index]

            score = float(royal.points) * 4.5

            if royal.ability:
                score += ABILITY_VALUE.get(royal.ability, 0.0)

            # Royal points can directly finish the game.
            if player.points + royal.points >= 20:
                score += 10_000.0

            return score

        return max(actions, key=royal_score)

    # ---------------------------------------------------------------------
    # DISCARD PHASE
    # ---------------------------------------------------------------------

    def _choose_discard(self, state: GameState, actions: list[Action]) -> Action:
        player = state.active
        target = self._best_target_card(state)

        if target is not None:
            missing = self._missing_after_tokens(player, target)

            def discard_cost(action: DiscardToken) -> float:
                gem = int(action.gem)

                # Keep gold whenever possible.
                if action.gem == Gem.GOLD:
                    return 100.0

                # Keep colours needed for the target.
                if gem < len(missing) and missing[gem] > 0:
                    return 20.0 + float(missing[gem])

                # Discard surplus. More abundant surplus is easier to discard.
                return -float(player.tokens[gem])

            return min(actions, key=discard_cost)

        def abundance(action: DiscardToken) -> float:
            if action.gem == Gem.GOLD:
                return -100.0
            return float(player.tokens[int(action.gem)])

        return max(actions, key=abundance)

    # ---------------------------------------------------------------------
    # SMALL HELPERS
    # ---------------------------------------------------------------------

    def _card_from_buy_action(self, state: GameState, action: BuyCard) -> Card:
        if action.source == "pyramid":
            return state.pyramid[action.level][action.index]
        return state.active.reserved[action.index]

    def _visible_and_reserved_cards(
            self,
            state: GameState,
            player: PlayerState,
    ) -> list[Card]:
        cards: list[Card] = []

        for level in (1, 2, 3):
            cards.extend(state.pyramid.get(level, []))

        cards.extend(player.reserved)

        return cards

    def _visible_and_reserved_cards_for_player_guess(
            self,
            state: GameState,
            player: PlayerState,
    ) -> list[Card]:
        """
        Same candidate pool, but used when estimating opponent plans.

        We cannot know opponent's private intentions, so this is only a guess.
        """
        return self._visible_and_reserved_cards(state, player)

    def _missing_after_tokens(
            self,
            player: PlayerState,
            card: Card,
    ) -> np.ndarray:
        """
        Missing coloured gems after permanent bonuses and current non-gold tokens.

        Gold is intentionally not subtracted here because it is flexible.
        Use sum(missing) <= gold to test affordability with gold.
        """
        cost = np.array(card.cost, dtype=np.int16)[:N_GEMS]
        bonuses = np.array(player.bonuses, dtype=np.int16)[:N_GEMS]
        tokens = np.array(player.tokens, dtype=np.int16)[:N_GEMS]

        return np.maximum(cost - bonuses - tokens, 0)

    def _gold_count(self, player: PlayerState) -> int:
        try:
            return int(player.tokens[Gem.GOLD])
        except Exception:
            return 0

    def _bonus_colour(self, card: Card) -> Optional[int]:
        """
        Return the colour index of a card bonus if it has one.

        Supports both vector-style gem_bonus and simple scalar-style values.
        """
        bonus = getattr(card, "gem_bonus", None)

        if bonus is None:
            return None

        try:
            arr = list(bonus)
            if not arr:
                return None
            idx = int(np.argmax(arr))
            if arr[idx] > 0:
                return idx
            return None
        except TypeError:
            try:
                return int(bonus)
            except Exception:
                return None

    def _card_would_win(self, player: PlayerState, card: Card) -> bool:
        """
        Estimate whether buying/taking this card immediately wins.
        """
        if player.points + card.points >= 20:
            return True

        if player.crowns + card.crowns >= 10:
            return True

        colour = self._bonus_colour(card)
        if colour is not None:
            mono_points = self._points_in_colour(player, colour) + card.points
            if mono_points >= 10:
                return True

        return False

    def _buy_wins_immediately(self, state: GameState, action: BuyCard) -> bool:
        card = self._card_from_buy_action(state, action)
        return self._card_would_win(state.active, card)

    def _victory_progress_bonus(self, player: PlayerState, card: Card) -> float:
        """
        Extra score for moving along one of the three victory tracks:
        - 20 total prestige,
        - 10 crowns,
        - 10 prestige in one colour.
        """
        bonus = 0.0

        new_points = player.points + card.points
        if new_points >= 20:
            bonus += 1000.0
        else:
            bonus += (new_points / 20.0) * float(card.points) * 1.5

        new_crowns = player.crowns + card.crowns
        if new_crowns >= 10:
            bonus += 1000.0
        else:
            bonus += (new_crowns / 10.0) * float(card.crowns) * 3.0

        colour = self._bonus_colour(card)
        if colour is not None and card.points > 0:
            new_mono = self._points_in_colour(player, colour) + card.points
            if new_mono >= 10:
                bonus += 1000.0
            else:
                bonus += (new_mono / 10.0) * float(card.points) * 1.8

        return bonus

    def _points_in_colour(self, player: PlayerState, colour: int) -> int:
        """
        Count prestige points on cards with the given bonus colour.

        This supports the 10-points-in-one-colour win condition.
        """
        total = 0

        for card in player.cards:
            if self._bonus_colour(card) == colour:
                total += int(card.points)

        return total
