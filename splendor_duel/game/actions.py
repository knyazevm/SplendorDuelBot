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


@dataclass(frozen=True)
class PassTurn:
    """
    End the turn without performing a main action.

    Legal ONLY when the player has no other action at all — you must move if
    you can.  The official rules do not cover this position, but it is
    reachable: privileges can strip the last non-gold tokens off the board
    while the bag is empty, the reserve is full and nothing is affordable.
    """
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
    """
    Assign the wildcard's bonus colour, which must be a colour already in
    the tableau.  The wildcard is always worth exactly 1 bonus of that
    colour, regardless of the target card's own bonus count.

    The action names the colour rather than a target card: the colour is all
    the engine needs, and two tableau cards of the same colour are the same
    decision.  Keying on the tableau index instead made the action space
    unbounded (a tableau can exceed 25 cards) and gave the network a slot
    whose meaning shifted with purchase order.
    """
    colour: int  # Gem index of the chosen bonus colour


@dataclass(frozen=True)
class EffectChooseGold:
    """
    Pick WHICH gold token to take when reserving a card.

    The tokens are interchangeable; the board cells they vacate are not.
    Vacating a gold changes no take-line immediately — a gold cell and an
    empty cell are equally non-takeable — but it decides where the next
    refill drops its tokens, which changes the legal take-set about half the
    time.  Only asked when the board holds more than one gold.
    """
    position: tuple[int, int]


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
        UseScroll | RefillBoard | ProceedToMain | PassTurn
        | TakeTokens | ReserveCard | BuyCard
        | EffectTakeSameGem | EffectTakeOpponentGem
        | EffectChooseWildcard | EffectChooseGold | EffectSkip
        | ChooseRoyal
        | DiscardToken
)
