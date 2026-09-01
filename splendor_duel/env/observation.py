"""
observation.py — Encode GameState into a flat numpy tensor.

The observation is always from the perspective of the ACTIVE player
(the one whose turn it is). This simplifies the network: it always
answers "what should I do?", never "what should player 0 do?".

Layout (all float32, mostly normalised to [0, 1]):

  Board:         5×5×8 = 200  (one-hot per cell: 7 gems + empty)
  My tokens:     7             (/10)
  My bonuses:    7             (/8)
  My stats:      5             (points/20, crowns/10, scrolls/3, n_cards/15, n_reserved/3)
  My royals:     4             (one-hot: which royals I have — by ability type)
  Opp tokens:    7             (/10)
  Opp bonuses:   7             (/8)
  Opp stats:     5
  Opp royals:    4
  Pyramid:       12 × 17 = 204 (cost[7] + bonus[7] + pts/6 + crowns/3 + is_wildcard)
  My reserved:   3 × 17 = 51
  Royals avail:  4 × 3 = 12   (present, points/3, ability_id/4)
  Meta:          6             (phase_onehot[5] + scrolls_center/3)
  ─────────────────────────────
  Total:         519

This is intentionally over-specified (redundant features) because
neural nets are good at ignoring irrelevant inputs, and having
a fixed-size observation simplifies everything.
"""
from __future__ import annotations

import numpy as np

from splendor_duel.game.constants import (
    BOARD_SIZE, Gem, GEM_NAMES, N_GEMS, PYRAMID_OPEN,
    VP_WIN, CROWNS_WIN, MAX_RESERVED,
)
from splendor_duel.game.actions import Phase
from splendor_duel.game.state import GameState

# ── Dimension constants ───────────────────────────────────────────────────────

_N_BOARD = BOARD_SIZE * BOARD_SIZE * 8  # 200
_N_TOKENS = N_GEMS  # 7
_N_BONUSES = N_GEMS  # 7
_N_STATS = 5
_N_ROYALS_PLAYER = 4
_N_PLAYER = _N_TOKENS + _N_BONUSES + _N_STATS + _N_ROYALS_PLAYER  # 23

_CARD_DIM = N_GEMS + N_GEMS + 1 + 1 + 1  # cost + bonus + pts + crowns + wildcard = 17
_MAX_PYRAMID = sum(PYRAMID_OPEN.values())  # 12
_N_PYRAMID = _MAX_PYRAMID * _CARD_DIM  # 204
_N_RESERVED = MAX_RESERVED * _CARD_DIM  # 51

_N_ROYALS_AVAIL = 4 * 3  # 12
_N_META = 6  # phase(5) + scrolls_center

OBS_SIZE = _N_BOARD + _N_PLAYER * 2 + _N_PYRAMID + _N_RESERVED + _N_ROYALS_AVAIL + _N_META
# 200 + 23*2 + 204 + 51 + 12 + 6 = 519


# ── Ability ID mapping ────────────────────────────────────────────────────────

_ABILITY_IDS = {
    None: 0,
    'extra_turn': 1,
    'take_same_gem': 2,
    'take_scroll': 3,
    'take_opponent_gem': 4,
}


# ── Public API ────────────────────────────────────────────────────────────────

def encode_state(state: GameState) -> np.ndarray:
    """
    Encode a GameState as a flat float32 vector of length OBS_SIZE.

    Perspective: always from active player's point of view.
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    off = 0

    # ── Board (one-hot) ──────────────────────────────────────
    board = state.board.grid
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = int(board[r, c])
            if v >= 0:
                obs[off + v] = 1.0
            else:
                obs[off + 7] = 1.0  # empty channel
            off += 8

    # ── Active player (me) ───────────────────────────────────
    me = state.active
    off = _encode_player(obs, off, me)

    # ── Opponent ─────────────────────────────────────────────
    opp = state.opponent
    off = _encode_player(obs, off, opp)

    # ── Pyramid cards ────────────────────────────────────────
    card_idx = 0
    for lvl in (1, 2, 3):
        cards = state.pyramid.get(lvl, [])
        for i in range(PYRAMID_OPEN[lvl]):
            if i < len(cards):
                off = _encode_card(obs, off, cards[i])
            else:
                off += _CARD_DIM  # empty slot (zeros)
            card_idx += 1

    # ── My reserved cards ────────────────────────────────────
    for i in range(MAX_RESERVED):
        if i < len(me.reserved):
            off = _encode_card(obs, off, me.reserved[i])
        else:
            off += _CARD_DIM

    # ── Royal cards available ────────────────────────────────
    for i in range(4):
        if i < len(state.royal_cards):
            r = state.royal_cards[i]
            obs[off] = 1.0  # present
            obs[off + 1] = r.points / 3.0  # normalised
            obs[off + 2] = _ABILITY_IDS.get(r.ability, 0) / 4.0
        off += 3

    # ── Meta ─────────────────────────────────────────────────
    # Phase one-hot (5 phases, excluding GAME_OVER)
    phase = state.phase
    if phase < 5:
        obs[off + int(phase)] = 1.0
    off += 5

    # Scrolls in center
    obs[off] = state.scrolls_center / 3.0
    off += 1

    assert off == OBS_SIZE, f"Observation size mismatch: {off} != {OBS_SIZE}"
    return obs


# ── Internal helpers ──────────────────────────────────────────────────────────

def _encode_player(obs: np.ndarray, off: int, player) -> int:
    """Encode player state into obs at offset. Returns new offset."""
    # Tokens (normalised by max 10)
    for i in range(N_GEMS):
        obs[off + i] = float(player.tokens[i]) / 10.0
    off += N_GEMS

    # Bonuses (normalised by ~8)
    for i in range(N_GEMS):
        obs[off + i] = float(player.bonuses[i]) / 8.0
    off += N_GEMS

    # Stats
    obs[off] = player.points / float(VP_WIN)  # /20
    obs[off + 1] = player.crowns / 10.0
    obs[off + 2] = player.scrolls / 3.0
    obs[off + 3] = len(player.cards) / 15.0
    obs[off + 4] = len(player.reserved) / float(MAX_RESERVED)
    off += 5

    # Royals held (one-hot by ability type)
    royal_vec = np.zeros(4, dtype=np.float32)
    for r in player.royals:
        aid = _ABILITY_IDS.get(r.ability, 0)
        if aid < 4:
            royal_vec[aid] = 1.0
    obs[off:off + 4] = royal_vec
    off += 4

    return off


CARD_ENC_SIZE = 2 * N_GEMS + 3

# card.id → its CARD_ENC_SIZE encoding. Cards are frozen dataclasses loaded
# once from cards.json, so a card's encoding never changes; building it with
# per-element numpy assignment on every call was ~6% of self-play time.
_CARD_ENC_CACHE: dict[str, np.ndarray] = {}


def _card_encoding(card) -> np.ndarray:
    """Return (and memoise) the fixed encoding of one card."""
    enc = _CARD_ENC_CACHE.get(card.id)
    if enc is None:
        enc = np.zeros(CARD_ENC_SIZE, dtype=np.float32)
        # Cost (normalised by 8)
        for i in range(N_GEMS):
            enc[i] = float(card.cost[i]) / 8.0
        # Bonus (normalised — double bonus cards have value 2); wildcards and
        # bonus-less cards leave this block at zero.
        if card.gem_bonus is not None:
            for i in range(N_GEMS):
                enc[N_GEMS + i] = float(card.gem_bonus[i]) / 2.0
        # Points, crowns, wildcard flag
        enc[2 * N_GEMS] = card.points / 6.0
        enc[2 * N_GEMS + 1] = card.crowns / 3.0
        enc[2 * N_GEMS + 2] = 1.0 if card.is_wildcard else 0.0
        _CARD_ENC_CACHE[card.id] = enc
    return enc


def _encode_card(obs: np.ndarray, off: int, card) -> int:
    """Encode a single card into obs at offset. Returns new offset."""
    obs[off:off + CARD_ENC_SIZE] = _card_encoding(card)
    return off + CARD_ENC_SIZE
