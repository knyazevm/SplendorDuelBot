"""
engine.py — Game engine: legal action generation and state transitions.

Turn flow:
    OPTIONAL ──▶ MAIN ──▶ EFFECT ──▶ ROYAL ──▶ DISCARD ──▶ next turn
    (scrolls,     (take/     (card       (pick     (drop to
     refill,       reserve/   effects)    royal)    ≤ 10)
     skip)         buy)

All public methods are static and return a new GameState (never mutate input).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .actions import (
    Action, BuyCard, ChooseRoyal, DiscardToken,
    EffectChooseWildcard, EffectSkip, EffectTakeOpponentGem,
    EffectTakeSameGem, Phase, ProceedToMain, RefillBoard,
    ReserveCard, TakeTokens, UseScroll,
)
from .board import Board
from .card import Card
from .constants import (
    ABILITY_EXTRA_TURN, ABILITY_TAKE_OPPONENT_GEM,
    ABILITY_TAKE_SAME_GEM, ABILITY_TAKE_SCROLL,
    GEM_NAMES, Gem, MAX_TOKENS, N_GEMS,
)
from .player import PlayerState
from .state import GameState


class GameEngine:
    """
    Pure-function game engine.

    Usage:
        state = GameState.new_game("data/cards.json")
        while not state.is_game_over:
            actions = GameEngine.get_legal_actions(state)
            action  = agent.choose(actions)
            state   = GameEngine.apply_action(state, action)
    """

    # ══════════════════════════════════════════════════════════════════════════
    # GET LEGAL ACTIONS
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def get_legal_actions(state: GameState) -> list[Action]:
        if state.phase == Phase.OPTIONAL:
            return _get_optional_actions(state)
        if state.phase == Phase.MAIN:
            return _get_main_actions(state)
        if state.phase == Phase.EFFECT:
            return _get_effect_actions(state)
        if state.phase == Phase.ROYAL:
            return _get_royal_actions(state)
        if state.phase == Phase.DISCARD:
            return _get_discard_actions(state)
        return []  # GAME_OVER

    # ══════════════════════════════════════════════════════════════════════════
    # APPLY ACTION
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def apply_action(state: GameState, action: Action) -> GameState:
        s = state.copy()

        if isinstance(action, UseScroll):
            return _apply_use_scroll(s, action)
        if isinstance(action, RefillBoard):
            return _apply_refill(s)
        if isinstance(action, ProceedToMain):
            s.phase = Phase.MAIN
            return s
        if isinstance(action, TakeTokens):
            return _apply_take_tokens(s, action)
        if isinstance(action, ReserveCard):
            return _apply_reserve(s, action)
        if isinstance(action, BuyCard):
            return _apply_buy(s, action)
        if isinstance(action, EffectChooseWildcard):
            return _apply_choose_wildcard(s, action)
        if isinstance(action, EffectTakeSameGem):
            return _apply_take_same_gem(s, action)
        if isinstance(action, EffectTakeOpponentGem):
            return _apply_take_opponent_gem(s, action)
        if isinstance(action, EffectSkip):
            return _advance_after_effect(s)
        if isinstance(action, ChooseRoyal):
            return _apply_choose_royal(s, action)
        if isinstance(action, DiscardToken):
            return _apply_discard(s, action)

        raise ValueError(f"Unknown action type: {type(action)}")


# ══════════════════════════════════════════════════════════════════════════════
# LEGAL ACTION GENERATORS (per phase)
# ══════════════════════════════════════════════════════════════════════════════

def _get_optional_actions(state: GameState) -> list[Action]:
    """
    Optional phase: use scrolls, refill board, or proceed to main.

    Rules order: scrolls first, then refill.  After refill → go to MAIN.
    Special case: if no MAIN actions are possible, RefillBoard is forced.
    """
    actions: list[Action] = []
    player = state.active

    # ── Use scroll: spend 1, take 1 non-gold token from board ────────────
    if player.scrolls > 0:
        positions = state.board.get_legal_single_positions(exclude_gold=True)
        for pos in positions:
            actions.append(UseScroll(position=pos))

    # ── Refill board: only if bag is not empty ────────────────────────────
    if sum(state.bag.values()) > 0:
        actions.append(RefillBoard())

    # ── Proceed to main: only if main actions exist ───────────────────────
    # Special rule: if no main actions are possible, must refill first.
    if _has_any_main_action(state):
        actions.append(ProceedToMain())

    # Edge case: if nothing at all is possible (empty board, empty bag,
    # no scrolls, no main actions) — shouldn't happen in normal play.
    # Fall through to ProceedToMain to avoid deadlock.
    if not actions:
        actions.append(ProceedToMain())

    return actions


def _get_main_actions(state: GameState) -> list[Action]:
    """Main phase: take tokens, reserve card, or buy card."""
    actions: list[Action] = []
    player = state.active

    # ── Take tokens (1–3 in a line, no gold) ──────────────────────────────
    for seg in state.board.get_legal_take_positions():
        actions.append(TakeTokens(positions=seg))

    # ── Reserve card (take 1 gold + reserve 1 card) ───────────────────────
    if player.can_reserve and state.board.has_gold():
        # From pyramid
        for lvl in (1, 2, 3):
            for idx in range(len(state.pyramid.get(lvl, []))):
                actions.append(ReserveCard(source='pyramid', level=lvl, index=idx))
        # From deck top (blind)
        for lvl in (1, 2, 3):
            if state.decks.get(lvl):
                actions.append(ReserveCard(source='deck', level=lvl, index=0))

    # ── Buy card ──────────────────────────────────────────────────────────
    # From pyramid
    for lvl in (1, 2, 3):
        for idx, card in enumerate(state.pyramid.get(lvl, [])):
            if _can_buy_card(player, card):
                actions.append(BuyCard(source='pyramid', level=lvl, index=idx))
    # From reserve
    for idx, card in enumerate(player.reserved):
        if _can_buy_card(player, card):
            actions.append(BuyCard(source='reserve', level=card.level, index=idx))

    return actions


def _get_effect_actions(state: GameState) -> list[Action]:
    """Effect phase: resolve the pending card effect."""
    effect = state.pending_effect
    player = state.active

    if effect == 'choose_wildcard':
        actions: list[Action] = []
        seen: set[tuple[int, int]] = set()
        for idx, card in enumerate(player.cards):
            bonus_info = _card_bonus_info(card, player)
            if bonus_info is not None:
                key = bonus_info  # (colour, count)
                if key not in seen:
                    seen.add(key)
                    actions.append(EffectChooseWildcard(target_card_index=idx))
        return actions if actions else [EffectSkip()]

    if effect == ABILITY_TAKE_SAME_GEM:
        card = state.pending_card
        assert card is not None
        colour = _card_bonus_colour(card, player)
        if colour is not None:
            positions = [
                pos for pos in state.board.get_legal_single_positions(exclude_gold=True)
                if state.board.token_at(*pos) == colour
            ]
            if positions:
                return [EffectTakeSameGem(position=pos) for pos in positions]
        return [EffectSkip()]

    if effect == ABILITY_TAKE_OPPONENT_GEM:
        opponent = state.opponent
        gems = [
            int(g) for g in range(N_GEMS)
            if g != Gem.GOLD and opponent.tokens[g] > 0
        ]
        if gems:
            return [EffectTakeOpponentGem(gem=g) for g in gems]
        return [EffectSkip()]

    # extra_turn and take_scroll are auto-applied, should not reach here
    return [EffectSkip()]


def _get_royal_actions(state: GameState) -> list[Action]:
    """Royal phase: choose one of the available royal cards."""
    return [
        ChooseRoyal(index=idx)
        for idx in range(len(state.royal_cards))
    ]


def _get_discard_actions(state: GameState) -> list[Action]:
    """Discard phase: choose one token to return to bag."""
    player = state.active
    return [
        DiscardToken(gem=int(g))
        for g in range(N_GEMS)
        if player.tokens[g] > 0
    ]


# ══════════════════════════════════════════════════════════════════════════════
# ACTION APPLICATION (each returns a new mutated-copy state)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_use_scroll(s: GameState, action: UseScroll) -> GameState:
    """Spend 1 scroll, take 1 non-gold token from board."""
    s.return_scroll_to_center(s.current_player)
    s.board, gem = s.board.take_single(*action.position)
    token_vec = np.zeros(N_GEMS, dtype=np.int8)
    token_vec[gem] = 1
    s.active.add_tokens(token_vec)
    # Stay in OPTIONAL for more scrolls or proceed
    return s


def _apply_refill(s: GameState) -> GameState:
    """Fill empty board cells from bag.  Opponent gets 1 scroll."""
    s.board, s.bag, _ = s.board.refill(s.bag)
    s.give_scroll_to(1 - s.current_player)
    # After refill → go to MAIN
    s.phase = Phase.MAIN
    return s


def _apply_take_tokens(s: GameState, action: TakeTokens) -> GameState:
    """Take 1–3 tokens in a contiguous line."""
    s.board, taken_vec = s.board.take_tokens(action.positions)
    s.active.add_tokens(taken_vec)

    # Trigger: 3 of same colour or 2 pearls → opponent gets scroll
    if Board.triggers_privilege(action.positions, taken_vec):
        s.give_scroll_to(1 - s.current_player)

    return _advance_after_main_no_buy(s)


def _apply_reserve(s: GameState, action: ReserveCard) -> GameState:
    """Take 1 gold from board + reserve 1 card."""
    # Take gold token from board
    gold_positions = [
        (r, c) for r in range(5) for c in range(5)
        if s.board.token_at(r, c) == Gem.GOLD
    ]
    # Pick any gold (choice doesn't matter, they're identical)
    s.board, _ = s.board.take_single(*gold_positions[0])
    gold_vec = np.zeros(N_GEMS, dtype=np.int8)
    gold_vec[Gem.GOLD] = 1
    s.active.add_tokens(gold_vec)

    # Take card
    if action.source == 'pyramid':
        card = s.pyramid[action.level][action.index]
        s.active.reserved.append(card)
        s.refill_pyramid_slot(action.level, action.index)
    else:  # deck
        card = s.decks[action.level].pop(0)
        s.active.reserved.append(card)

    return _advance_after_main_no_buy(s)


def _apply_buy(s: GameState, action: BuyCard) -> GameState:
    """Buy a card: pay tokens, gain card (or defer for wildcard/effect)."""
    # Get the card
    if action.source == 'pyramid':
        card = s.pyramid[action.level][action.index]
    else:  # reserve
        card = s.active.reserved[action.index]

    # Compute and pay
    payment = s.active.compute_payment(card)
    assert payment is not None, f"Cannot afford card {card.id}"
    s.active.remove_tokens(payment)

    # Return payment to bag
    for i in range(N_GEMS):
        s.bag[GEM_NAMES[i]] += int(payment[i])

    # Remove card from source
    if action.source == 'pyramid':
        s.refill_pyramid_slot(action.level, action.index)
    else:
        s.active.reserved.pop(action.index)

    # ── Handle card addition and effects ──────────────────────────────────
    if card.is_wildcard:
        # Defer: need player to choose target card
        s.pending_card = card
        s.pending_effect = 'choose_wildcard'
        s.phase = Phase.EFFECT
        return s

    # Non-wildcard: add card immediately
    s.active.add_card(card)
    s.pending_card = card

    # Check for card ability
    return _handle_card_ability(s, card)


def _apply_choose_wildcard(s: GameState, action: EffectChooseWildcard) -> GameState:
    """Resolve wildcard: place on target card, copy its bonus."""
    card = s.pending_card
    assert card is not None and card.is_wildcard

    target = s.active.cards[action.target_card_index]
    bonus_info = _card_bonus_info(target, s.active)
    assert bonus_info is not None, f"Target card {target.id} has no bonus"

    colour, count = bonus_info
    s.active.add_card(card, wildcard_color=colour, wildcard_count=count)

    # Handle the ability (if any)
    return _handle_card_ability(s, card)


def _apply_take_same_gem(s: GameState, action: EffectTakeSameGem) -> GameState:
    """Take 1 matching-colour token from board."""
    s.board, gem = s.board.take_single(*action.position)
    token_vec = np.zeros(N_GEMS, dtype=np.int8)
    token_vec[gem] = 1
    s.active.add_tokens(token_vec)
    return _advance_after_effect(s)


def _apply_take_opponent_gem(s: GameState, action: EffectTakeOpponentGem) -> GameState:
    """Take 1 non-gold token from opponent."""
    gem = action.gem
    assert gem != Gem.GOLD
    remove_vec = np.zeros(N_GEMS, dtype=np.int8)
    remove_vec[gem] = 1
    s.opponent.remove_tokens(remove_vec)
    s.active.add_tokens(remove_vec)
    return _advance_after_effect(s)


def _apply_choose_royal(s: GameState, action: ChooseRoyal) -> GameState:
    """Pick a royal card, apply its ability."""
    royal = s.royal_cards.pop(action.index)
    s.active.add_royal(royal)

    # Check if another royal needed first (e.g. jumped 2→6 crowns)
    if s.active.needs_royal() and s.royal_cards:
        s.phase = Phase.ROYAL
        return s

    # Apply royal card ability
    if royal.ability == ABILITY_TAKE_SCROLL:
        s.give_scroll_to(s.current_player)
        return _advance_after_royal(s)

    if royal.ability == ABILITY_EXTRA_TURN:
        s.extra_turn_flag = True
        return _advance_after_royal(s)

    if royal.ability in (ABILITY_TAKE_SAME_GEM, ABILITY_TAKE_OPPONENT_GEM):
        # Need player input — enter EFFECT phase
        s.pending_effect = royal.ability
        s.pending_card = None
        s.phase = Phase.EFFECT
        effect_actions = _get_effect_actions(s)
        if len(effect_actions) == 1 and isinstance(effect_actions[0], EffectSkip):
            s.pending_effect = None
            return _advance_after_royal(s)
        return s

    return _advance_after_royal(s)


def _apply_discard(s: GameState, action: DiscardToken) -> GameState:
    """Discard 1 token to bag."""
    remove_vec = np.zeros(N_GEMS, dtype=np.int8)
    remove_vec[action.gem] = 1
    s.active.remove_tokens(remove_vec)
    s.bag[GEM_NAMES[action.gem]] += 1

    if s.active.tokens_over_limit > 0:
        return s  # keep discarding

    return _advance_after_discard(s)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE TRANSITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _handle_card_ability(s: GameState, card: Card) -> GameState:
    """After adding a non-wildcard card, handle its ability (as well as wildcards after handling wildness)."""
    ability = card.ability

    if ability is None:
        return _advance_after_effect(s)

    # Auto-apply abilities that require no input
    if ability == ABILITY_EXTRA_TURN:
        s.extra_turn_flag = True
        return _advance_after_effect(s)

    if ability == ABILITY_TAKE_SCROLL:
        s.give_scroll_to(s.current_player)
        return _advance_after_effect(s)

    # Abilities that need player input → EFFECT phase
    if ability in (ABILITY_TAKE_SAME_GEM, ABILITY_TAKE_OPPONENT_GEM):
        s.pending_effect = ability
        s.phase = Phase.EFFECT
        # Check if effect actually has targets; if not, auto-skip
        effect_actions = _get_effect_actions(s)
        if len(effect_actions) == 1 and isinstance(effect_actions[0], EffectSkip):
            s.pending_effect = None
            return _advance_after_effect(s)
        return s

    # Unknown ability — skip
    return _advance_after_effect(s)


def _advance_after_main_no_buy(s: GameState) -> GameState:
    """After TakeTokens or ReserveCard (no card bought)."""
    s.pending_card = None
    s.pending_effect = None
    # No card bought → no effects, no royal check.
    # Just check token limit.
    if s.active.tokens_over_limit > 0:
        s.phase = Phase.DISCARD
        return s
    return _end_turn(s)


def _advance_after_effect(s: GameState) -> GameState:
    """After all card effects are resolved."""
    s.pending_effect = None

    # ── Royal card check ──────────────────────────────────────────────────
    if s.active.needs_royal() and s.royal_cards:
        s.phase = Phase.ROYAL
        return s

    return _advance_after_royal(s)


def _advance_after_royal(s: GameState) -> GameState:
    """After royal card phase (or if skipped)."""
    # ── Token limit ───────────────────────────────────────────────────────
    if s.active.tokens_over_limit > 0:
        s.phase = Phase.DISCARD
        return s

    return _advance_after_discard(s)


def _advance_after_discard(s: GameState) -> GameState:
    """After discarding down to ≤ 10 tokens."""
    # ── Victory check ─────────────────────────────────────────────────────
    if s.active.check_victory() is not None:
        s.phase = Phase.GAME_OVER
        return s

    return _end_turn(s)


def _end_turn(s: GameState) -> GameState:
    """Switch to next player (or repeat if extra_turn)."""
    s.pending_card = None
    s.pending_effect = None

    if s.extra_turn_flag:
        s.extra_turn_flag = False
        s.phase = Phase.OPTIONAL
        s.turn += 1
        return s

    # Normal turn end → switch player
    s.current_player = 1 - s.current_player
    s.phase = Phase.OPTIONAL
    s.turn += 1
    return s


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _has_any_main_action(state: GameState) -> bool:
    """
    Quick check whether ANY main action is possible.
    Used in OPTIONAL phase to decide if ProceedToMain is available.
    """
    player = state.active

    # Can take tokens?
    if state.board.get_legal_take_positions():
        return True

    # Can reserve?
    if player.can_reserve and state.board.has_gold():
        return True

    # Can buy any card?
    for lvl in (1, 2, 3):
        for card in state.pyramid.get(lvl, []):
            if _can_buy_card(player, card):
                return True
    for card in player.reserved:
        if _can_buy_card(player, card):
            return True

    return False


def _can_buy_card(player: PlayerState, card: Card) -> bool:
    """Check if player can buy this card (affordability + wildcard requirement)."""
    if card.is_wildcard and not player.has_bonuses:
        return False
    return player.can_afford(card)


def _card_bonus_colour(card: Card, player: PlayerState) -> Optional[int]:
    """
    Return the primary bonus colour of a card.
    For wildcard cards, returns the assigned colour.
    """
    if card.is_wildcard:
        return player.wildcard_assignments.get(card.id)
    if card.gem_bonus is not None:
        for i, v in enumerate(card.gem_bonus):
            if v > 0:
                return i
    return None


def _card_bonus_info(card: Card, player: PlayerState) -> Optional[tuple[int, int]]:
    """
    Return (colour, count) of a card's bonus, or None if no bonus.
    For wildcard cards, returns the assigned bonus info.
    """
    if card.is_wildcard:
        colour = player.wildcard_assignments.get(card.id)
        if colour is None:
            return None
        # Wildcard always gives 1 of the assigned colour
        return colour, 1
    if card.gem_bonus is not None:
        for i, v in enumerate(card.gem_bonus):
            if v > 0:
                return i, int(v)
    return None
