"""
card.py — Immutable Card and RoyalCard dataclasses + JSON loader.

Design notes:
- frozen=True  → hashable, safe as dict keys and in sets
- cost / bonus stored as numpy-ready tuples (length N_GEMS)
  so callers can do:  np.array(card.cost) without any conversion
- wildcard cards store is_wildcard=True and gem_bonus=None;
  their effective bonus is resolved at runtime against the tableau
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import GEM_NAMES, N_GEMS


# ── Card ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Card:
    id: str
    level: int  # 1 / 2 / 3
    cost: tuple[int, ...]  # length N_GEMS, indexed by Gem enum
    gem_bonus: Optional[tuple[int, ...]]  # length N_GEMS, or None (wildcard / no bonus)
    is_wildcard: bool  # True → bonus colour chosen at buy-time
    points: int
    crowns: int
    ability: Optional[str]

    # ── Convenience numpy views (not stored, computed on access) ──────────────

    @property
    def cost_vec(self) -> np.ndarray:
        """Cost as int8 vector of length N_GEMS."""
        return np.array(self.cost, dtype=np.int8)

    @property
    def bonus_vec(self) -> np.ndarray:
        """
        Permanent bonus as int8 vector.
        Wildcards return zeros until resolved against the tableau.
        """
        if self.gem_bonus is None:
            return np.zeros(N_GEMS, dtype=np.int8)
        return np.array(self.gem_bonus, dtype=np.int8)

    def __repr__(self) -> str:
        bonus_str = 'wildcard' if self.is_wildcard else str(self.gem_bonus)
        return (
            f"Card({self.id} L{self.level} "
            f"pts={self.points} cr={self.crowns} "
            f"bonus={bonus_str} abil={self.ability})"
        )


# ── RoyalCard ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoyalCard:
    id: str
    points: int  # 2 or 3
    ability: Optional[str]

    def __repr__(self) -> str:
        return f"Royal({self.id} pts={self.points} abil={self.ability})"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cost_to_tuple(cost_dict: dict) -> tuple[int, ...]:
    """Convert {color: count} dict to a fixed-length tuple indexed by Gem."""
    return tuple(int(cost_dict.get(name, 0)) for name in GEM_NAMES)


def _bonus_to_parts(
        gem_bonus: Optional[dict],
) -> tuple[Optional[tuple[int, ...]], bool]:
    """
    Parse gem_bonus JSON field.

    Returns (bonus_tuple_or_None, is_wildcard).
    - wildcard  → (None, True)
    - real gem  → (tuple, False)
    - no bonus  → (None, False)
    """
    if gem_bonus is None:
        return None, False
    if 'wildcard' in gem_bonus:
        return None, True
    vec = [0] * N_GEMS
    for name, val in gem_bonus.items():
        vec[GEM_NAMES.index(name)] = int(val)
    return tuple(vec), False


# ── Public loader ─────────────────────────────────────────────────────────────

def load_cards(path: str) -> tuple[list[Card], list[RoyalCard]]:
    """
    Load cards.json and return (cards, royal_cards).

    Expected JSON structure:
    {
        "cards":       [{id, level, cost, gem_bonus, points, crowns, ability}, ...],
        "royal_cards": [{id, points, ability}, ...]
    }
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    cards: list[Card] = []
    for raw in data['cards']:
        bonus_tuple, is_wc = _bonus_to_parts(raw.get('gem_bonus'))
        cards.append(Card(
            id=raw['id'],
            level=int(raw['level']),
            cost=_cost_to_tuple(raw['cost']),
            gem_bonus=bonus_tuple,
            is_wildcard=is_wc,
            points=int(raw.get('points', 0)),
            crowns=int(raw.get('crowns', 0)),
            ability=raw.get('ability'),
        ))

    royal_cards: list[RoyalCard] = []
    for raw in data['royal_cards']:
        royal_cards.append(RoyalCard(
            id=raw['id'],
            points=int(raw.get('points', 2)),
            ability=raw.get('ability'),
        ))

    return cards, royal_cards
