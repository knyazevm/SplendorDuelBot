"""
actions.py — Immutable action types for every decision point in a turn.

A turn flows through phases:

    OPTIONAL ──▶ MAIN ──▶ EFFECT ──▶ ROYAL ──▶ DISCARD ──▶ end turn
    (scroll,      (take/     (card       (pick     (drop
     refill,       reserve/   effects)    royal)    tokens)
     skip)         buy)

Each phase has its own set of legal actions.
All actions are frozen (hashable, safe for MCTS).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


# ── Turn phases ───────────────────────────────────────────────────────────────

class Phase(IntEnum):
    OPTIONAL = 0  # can use scrolls, refill board, or proceed to MAIN
    MAIN = 1  # must take tokens / reserve / buy
    EFFECT = 2  # must resolve bought card's effect
    ROYAL = 3  # must choose a royal card
    DISCARD = 4  # must discard tokens (one at a time, until ≤ 10)
    GAME_OVER = 5


# ── Optional-phase actions ────────────────────────────────────────────────────

@dataclass(frozen=True)
class UseScroll:
    """Spend 1 scroll → take 1 non-gold token from the board."""
    position: tuple[int, int]  # (row, col) of token to take


@dataclass(frozen=True)
class RefillBoard:
    """Fill empty board cells from the bag.  Opponent gets 1 scroll."""
    pass


@dataclass(frozen=True)
class ProceedToMain:
    """Skip remaining optional actions, go to MAIN phase."""
    pass


# ── Main-phase actions ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TakeTokens:
    """Take 1–3 tokens in a contiguous line (no gold)."""
    positions: tuple[tuple[int, int], ...]  # board cells to take


@dataclass(frozen=True)
class ReserveCard:
    """Take 1 gold from board + reserve 1 card (pyramid or deck top)."""
    source: str  # 'pyramid' or 'deck'
    level: int  # 1, 2, 3
    index: int  # position within pyramid row (ignored for 'deck')


@dataclass(frozen=True)
class BuyCard:
    """Buy 1 card from pyramid or reserve.  Payment computed by engine."""
    source: str  # 'pyramid' or 'reserve'
    level: int  # 1, 2, 3  (ignored when source='reserve')
    index: int  # position within pyramid row or reserve list


# ── Effect-phase actions ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class EffectTakeSameGem:
    """take_same_gem: pick a matching-colour token from the board."""
    position: tuple[int, int]


@dataclass(frozen=True)
class EffectTakeOpponentGem:
    """take_opponent_gem: pick 1 non-gold token from opponent."""
    gem: int  # Gem index


@dataclass(frozen=True)
class EffectChooseWildcard:
    """Place wildcard on an existing card to copy its bonus colour & count."""
    target_card_index: int  # index into player.cards (must have bonus)


@dataclass(frozen=True)
class EffectSkip:
    """Effect cannot be applied (no valid targets).  Auto-skip."""
    pass


# ── Royal-phase action ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChooseRoyal:
    """Pick one of the available royal cards."""
    index: int  # position in the royal cards list


# ── Discard-phase action ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DiscardToken:
    """Discard 1 token to get back to ≤ 10."""
    gem: int  # Gem index to discard


# ── Type alias for convenience ────────────────────────────────────────────────

Action = (
        UseScroll | RefillBoard | ProceedToMain
        | TakeTokens | ReserveCard | BuyCard
        | EffectTakeSameGem | EffectTakeOpponentGem
        | EffectChooseWildcard | EffectSkip
        | ChooseRoyal
        | DiscardToken
)
