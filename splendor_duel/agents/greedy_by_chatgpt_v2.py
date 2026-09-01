"""
greedy_by_chatgpt.py — stronger heuristic agent for Splendor Duel.

Version: v2

Design goals:
- Prefer immediate wins.
- Prefer buying cards aggressively.
- Track crowns and mono-colour prestige as secondary win paths.
- Take tokens that make strong cards affordable soon.
- Avoid giving opponent privileges unless the move is clearly worth it.
- Reserve cards only when the reserve is clearly valuable.
- Handle effects, royal choices, and discard decisions with the same target logic.

This is NOT a learning model.
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
    "extra_turn": 4.5,
    "take_opponent_gem": 2.6,
    "take_same_gem": 2.1,
    "take_scroll": 1.6,
}


class GreedyByChatGPTV2(BaseAgent):
    """
    A stronger greedy baseline.

    Main priorities:
    1. Win immediately if possible.
    2. Buy the best available card.
    3. Take tokens that create the strongest next-buy opportunity.
    4. Reserve only if it is clearly better than taking tokens.
    5. Block opponent's immediate wins when possible.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(name="GreedyByChatGPT_v2")
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

        # 2. Buy almost always.
        # Tournament result suggests that tempo matters more than over-planning.
        if buy_actions:
            return max(buy_actions, key=lambda a: self._buy_score(state, a))

        target = self._best_target_card(state)

        best_take: Optional[TakeTokens] = None
        take_score = -10_000.0

        if take_actions:
            best_take = max(
                take_actions,
                key=lambda a: self._take_tokens_score(state, a, target),
            )
            take_score = self._take_tokens_score(state, best_take, target)

        best_reserve: Optional[ReserveCard] = None
        reserve_score = -10_000.0

        if reserve_actions:
            best_reserve = max(
                reserve_actions,
                key=lambda a: self._reserve_score(state, a),
            )
            reserve_score = self._reserve_score(state, best_reserve)

        # Reserve only if it is clearly better.
        # This prevents losing tempo by reserving too often.
        if best_reserve is not None and reserve_score > take_score + 1.2:
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
            score += self._opponent_denial_value(opponent, card) * 0.20

        return score

    def _card_intrinsic_score(self, card: Card, player: PlayerState) -> float:
        """
        Raw card value.

        v2 tuning:
        - More weight on prestige points.
        - Crowns are useful but should not distract too much.
        - Abilities are valuable because they create tempo.
        - Permanent bonuses matter because they accelerate future buys.
        """
        score = 0.0

        score += float(card.points) * 6.2
        score += float(card.crowns) * 2.7

        if card.ability:
            score += ABILITY_VALUE.get(card.ability, 0.0)

        bonus_colour = self._bonus_colour(card)
        if bonus_colour is not None:
            score += 1.15

            try:
                # First few bonuses of a colour are more valuable than later ones.
                score += max(0.0, 2.5 - float(player.bonuses[bonus_colour])) * 0.25
            except Exception:
                pass

        if getattr(card, "is_wildcard", False):
            score += 1.4

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
        Pick a realistic near-term target.

        v2:
        - Strongly prefer cards that are close.
        - Penalize far-away cards even if they are theoretically valuable.
        - Still notice immediate victory routes.
        """
        player = state.active

        best_card: Optional[Card] = None
        best_score = -10_000.0

        for card in self._visible_and_reserved_cards(state, player):
            missing = self._missing_after_tokens(player, card)
            distance = int(missing.sum())
            gold = self._gold_count(player)
            effective_distance = max(0, distance - gold)

            intrinsic = self._card_intrinsic_score(card, player)
            progress = self._victory_progress_bonus(player, card)

            # Close cards should dominate the target choice.
            score = intrinsic / (effective_distance + 1)

            # Small strategic bonus, not too large.
            score += progress * 0.18

            # Big penalty for distant dreams.
            if effective_distance >= 6:
                score *= 0.55
            elif effective_distance >= 4:
                score *= 0.75

            # Winning card remains absolute priority.
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

        # More tokens are good, but not as important as usefulness.
        score += len(action.positions) * 0.35

        # Diversity usually improves flexibility.
        score += int(np.count_nonzero(taken)) * 0.25

        if target is not None:
            missing = self._missing_after_tokens(player, target)
            useful = np.minimum(taken, missing)

            # Closing target gap.
            score += float(useful.sum()) * 3.7

            after_missing = np.maximum(missing - taken, 0)

            # Huge tempo bonus: this move makes target affordable.
            if int(after_missing.sum()) <= self._gold_count(player):
                score += 6.5

            # Mild penalty for irrelevant tokens.
            waste = np.maximum(taken - missing, 0)
            score -= float(waste.sum()) * 0.30

        # Extra: does this token action make any good card affordable next?
        score += self._next_buy_potential_after_take(state, taken) * 0.85

        # Avoid going too far over token limit.
        try:
            current_tokens = int(np.array(player.tokens).sum())
            after_tokens = current_tokens + len(action.positions)
            overflow = max(0, after_tokens - 10)
            score -= overflow * 1.4
        except Exception:
            pass

        # Giving opponent privilege is bad, but can be worth it for strong tempo.
        if Board.triggers_privilege(action.positions, taken):
            score -= 2.0

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

    def _next_buy_potential_after_take(
            self,
            state: GameState,
            taken: np.ndarray,
    ) -> float:
        """
        Estimate how much this token action improves next-turn buying options.

        This is important because strong greedy play is often:
            take exactly the tokens that make a strong card affordable next.
        """
        player = state.active
        best = 0.0

        for card in self._visible_and_reserved_cards(state, player):
            before_missing = self._missing_after_tokens(player, card)
            after_missing = np.maximum(before_missing - taken, 0)

            before_distance = max(
                0,
                int(before_missing.sum()) - self._gold_count(player),
            )
            after_distance = max(
                0,
                int(after_missing.sum()) - self._gold_count(player),
            )

            improvement = before_distance - after_distance
            if improvement <= 0:
                continue

            value = self._card_intrinsic_score(card, player)

            local_score = improvement * 0.8

            # If the card becomes affordable, this is the main prize.
            if after_distance == 0:
                local_score += value * 0.45

                if self._card_would_win(player, card):
                    local_score += 500.0

            # Prefer improving already-near cards.
            local_score += value / (after_distance + 2) * 0.15

            if local_score > best:
                best = local_score

        return best

    # ---------------------------------------------------------------------
    # RESERVE SCORING
    # ---------------------------------------------------------------------

    def _reserve_score(self, state: GameState, action: ReserveCard) -> float:
        player = state.active
        opponent = state.opponent

        if action.source == "deck":
            # Blind reserve is usually slow. Keep it modest.
            return 0.7 + float(action.level) * 0.45

        card = state.pyramid[action.level][action.index]

        own_eff = self._card_efficiency(card, player)
        own_value = own_eff * 0.55

        denial = self._opponent_denial_value(opponent, card)

        score = own_value + denial * 0.35

        # Reserving an opponent's immediate win is critical.
        if self._card_would_win(opponent, card):
            score += 1000.0

        # Reserving an opponent's already-affordable high-value card is useful.
        opp_missing = self._missing_after_tokens(opponent, card)
        if int(opp_missing.sum()) <= self._gold_count(opponent):
            score += self._card_intrinsic_score(card, opponent) * 0.25

        # Gold is useful, but not enough to justify weak reserves.
        target = self._best_target_card(state)
        if target is not None:
            missing = self._missing_after_tokens(player, target)
            if int(missing.sum()) > self._gold_count(player):
                score += 0.8

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

                    gem_i = int(gem)
                    score = 0.0

                    if missing[gem_i] > 0:
                        score += 4.0 + float(missing[gem_i])

                    # Extra value if this scroll makes the target affordable.
                    after_missing = missing.copy()
                    after_missing[gem_i] = max(0, after_missing[gem_i] - 1)

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
                    EffectTakeSameGem identifies the token by board position
                    in this project, not by a direct .gem attribute.
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
                for card in self._visible_and_reserved_cards_for_player_guess(
                        state,
                        opponent,
                ):
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
        return float(missing[action.colour]) + 0.25

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
        Extra score for victory progress.

        v2:
        - Immediate wins are huge.
        - Non-immediate mono/crown progress is intentionally moderate.
        - Prestige remains the default main plan.
        """
        bonus = 0.0

        new_points = player.points + card.points

        if new_points >= 20:
            bonus += 1000.0
        else:
            bonus += (new_points / 20.0) * float(card.points) * 1.8

        new_crowns = player.crowns + card.crowns

        if new_crowns >= 10:
            bonus += 1000.0
        else:
            bonus += (new_crowns / 10.0) * float(card.crowns) * 1.8

        colour = self._bonus_colour(card)

        if colour is not None and card.points > 0:
            new_mono = self._points_in_colour(player, colour) + card.points

            if new_mono >= 10:
                bonus += 1000.0
            else:
                bonus += (new_mono / 10.0) * float(card.points) * 0.75

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
