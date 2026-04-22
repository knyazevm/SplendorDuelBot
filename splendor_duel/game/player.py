"""
player.py — Per-player state.

Design:
- Internally mutable (numpy arrays + lists) for speed in engine.
- copy() produces a fully independent clone — required for MCTS.
- Affordability checks are vectorised: cost - bonuses - tokens → gold needed.
- "In the game there are no pearl bonuses" — bonuses[PEARL] is always 0.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np

from .card import Card, RoyalCard
from .constants import (
    CROWNS_ROYAL_1,
    CROWNS_ROYAL_2,
    CROWNS_WIN,
    GEM_NAMES,
    Gem,
    MAX_RESERVED,
    MAX_TOKENS,
    MONO_VP_WIN,
    N_GEMS,
    VP_WIN,
)


class PlayerState:
    __slots__ = (
        'tokens', 'bonuses', 'cards', 'reserved', 'scrolls',
        'crowns', 'points', 'royals', 'wildcard_assignments',
    )

    def __init__(self) -> None:
        self.tokens: np.ndarray = np.zeros(N_GEMS, dtype=np.int8)
        self.bonuses: np.ndarray = np.zeros(N_GEMS, dtype=np.int8)
        self.cards: list[Card] = []
        self.reserved: list[Card] = []
        self.scrolls: int = 0
        self.crowns: int = 0
        self.points: int = 0
        self.royals: list[RoyalCard] = []
        # card_id → assigned Gem index (for wildcard cards)
        self.wildcard_assignments: dict[str, int] = {}

    # ── Copy ──────────────────────────────────────────────────────────────────

    def copy(self) -> PlayerState:
        p = PlayerState.__new__(PlayerState)
        p.tokens = self.tokens.copy()
        p.bonuses = self.bonuses.copy()
        p.cards = list(self.cards)  # Cards are frozen — shallow OK
        p.reserved = list(self.reserved)
        p.scrolls = self.scrolls
        p.crowns = self.crowns
        p.points = self.points
        p.royals = list(self.royals)
        p.wildcard_assignments = dict(self.wildcard_assignments)
        return p

    # ── Token helpers ─────────────────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return int(self.tokens.sum())

    @property
    def tokens_over_limit(self) -> int:
        return max(0, self.total_tokens - MAX_TOKENS)

    def add_tokens(self, vec: np.ndarray) -> None:
        self.tokens = self.tokens + vec

    def remove_tokens(self, vec: np.ndarray) -> None:
        assert np.all(self.tokens >= vec), \
            f"Not enough tokens: have {self.tokens}, removing {vec}"
        self.tokens = self.tokens - vec

    # ── Card / bonus helpers ──────────────────────────────────────────────────

    def add_card(
            self,
            card: Card,
            wildcard_color: Optional[int] = None,
            wildcard_count: int = 1,
    ) -> None:
        """
        Add a bought card: update cards list, bonuses, crowns, points.
        For wildcard cards, wildcard_color and wildcard_count must be provided.
        wildcard_count copies the bonus count of the card the wildcard is placed on.
        """
        self.cards.append(card)
        self.points += card.points
        self.crowns += card.crowns

        if card.is_wildcard:
            assert wildcard_color is not None, "Must assign colour for wildcard"
            self.wildcard_assignments[card.id] = wildcard_color
            self.bonuses[wildcard_color] += wildcard_count
        elif card.gem_bonus is not None:
            self.bonuses += np.array(card.gem_bonus, dtype=np.int8)

    def add_royal(self, royal: RoyalCard) -> None:
        self.royals.append(royal)
        self.points += royal.points

    @property
    def can_reserve(self) -> bool:
        return len(self.reserved) < MAX_RESERVED

    @property
    def has_bonuses(self) -> bool:
        """True if player has at least one card with a bonus (needed for wildcard buys)."""
        return int(self.bonuses.sum()) > 0

    @property
    def n_royals(self) -> int:
        return len(self.royals)

    def needs_royal(self) -> bool:
        """True if the player just crossed a crown threshold and must pick a royal."""
        if self.n_royals == 0 and self.crowns >= CROWNS_ROYAL_1:
            return True
        if self.n_royals == 1 and self.crowns >= CROWNS_ROYAL_2:
            return True
        return False

    # ── Affordability ─────────────────────────────────────────────────────────

    def can_afford(self, card: Card) -> bool:
        """Check if the player can buy this card (with gold substitution)."""
        return self.compute_payment(card) is not None

    def compute_payment(self, card: Card) -> Optional[np.ndarray]:
        """
        Compute the token payment for a card, or None if unaffordable.

        Returns an int8 vector (length N_GEMS) of tokens to spend.
        Gold tokens fill any shortfall in specific colours.
        """
        cost = np.array(card.cost, dtype=np.int8)
        # Reduce cost by permanent bonuses (bonuses never include pearl or gold)
        needed = np.maximum(cost - self.bonuses, 0)
        # Pay with same-colour tokens (excluding gold slot for now)
        paid = np.minimum(needed, self.tokens)
        # Gold covers the rest
        shortfall = needed - paid
        # Gold needed (for all non-gold gem types)
        gold_needed = int(shortfall[:Gem.GOLD].sum())  # indices 0..5
        # Pearl shortfall is also covered by gold
        # (shortfall already includes pearl at index PEARL)
        if gold_needed > self.tokens[Gem.GOLD]:
            return None  # can't afford
        payment = paid.copy()
        payment[Gem.GOLD] = gold_needed
        return payment

    # ── Victory conditions ────────────────────────────────────────────────────

    def check_victory(self) -> Optional[str]:
        """
        Check all three victory conditions.
        Returns a string describing the condition, or None.
        """
        # Condition 1: ≥ 20 prestige points total
        if self.points >= VP_WIN:
            return 'prestige_20'

        # Condition 2: ≥ 10 crowns
        if self.crowns >= CROWNS_WIN:
            return 'crowns_10'

        # Condition 3: ≥ 10 prestige on cards sharing one bonus colour
        mono = self._mono_colour_points()
        for colour, pts in mono.items():
            if pts >= MONO_VP_WIN:
                return f'mono_{GEM_NAMES[colour]}'

        return None

    def _mono_colour_points(self) -> dict[int, int]:
        """
        Sum prestige points per bonus colour.

        Rules for condition 3:
        - Only cards WITH bonuses count.
        - Wildcard cards count toward the colour they were assigned to.
        - A card with double bonus (e.g. black:2) still counts once for
          that colour (it's one card, one colour).
        """
        result: dict[int, int] = {}
        for card in self.cards:
            if card.is_wildcard:
                colour = self.wildcard_assignments.get(card.id)
                if colour is not None:
                    result[colour] = result.get(colour, 0) + card.points
            elif card.gem_bonus is not None:
                # Find the colour(s) this card provides bonus for
                for i, v in enumerate(card.gem_bonus):
                    if v > 0:
                        result[i] = result.get(i, 0) + card.points
                        break  # each card has one colour (even if 2 bonuses)
        return result

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        tok_str = ', '.join(f'{GEM_NAMES[i]}={self.tokens[i]}'
                            for i in range(N_GEMS) if self.tokens[i] > 0)
        bon_str = ', '.join(f'{GEM_NAMES[i]}={self.bonuses[i]}'
                            for i in range(N_GEMS) if self.bonuses[i] > 0)
        return (
            f"Player(pts={self.points} cr={self.crowns} scr={self.scrolls} "
            f"tok=[{tok_str}] bon=[{bon_str}] "
            f"cards={len(self.cards)} res={len(self.reserved)} "
            f"royals={self.n_royals})"
        )
