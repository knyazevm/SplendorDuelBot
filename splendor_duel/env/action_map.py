"""
action_map.py — Fixed action space for Gymnasium.

Maps every possible game Action to a unique integer index [0, N_ACTIONS).
Provides:
  - action_to_index(action) → int
  - index_to_action(index, state) → Action
  - legal_mask(state) → np.bool array of length N_ACTIONS
"""
from __future__ import annotations

import numpy as np

from splendor_duel.game.actions import (
    Action, BuyCard, ChooseRoyal, DiscardToken,
    EffectChooseWildcard, EffectSkip, EffectTakeOpponentGem,
    EffectTakeSameGem, ProceedToMain, RefillBoard,
    ReserveCard, TakeTokens, UseScroll,
)
from splendor_duel.game.constants import (
    ALL_SEGMENTS, BOARD_SIZE, Gem, N_GEMS, PYRAMID_OPEN, MAX_RESERVED,
)
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState

# ── Build static index tables ─────────────────────────────────────────────────

# Segment → index for TakeTokens
_SEG_TO_IDX: dict[tuple[tuple[int, int], ...], int] = {
    seg: i for i, seg in enumerate(ALL_SEGMENTS)
}
_N_TAKE = len(ALL_SEGMENTS)  # 145

# Pyramid slots: (level, index) → offset
_PYRAMID_SLOTS: list[tuple[int, int]] = []
for lvl in (1, 2, 3):
    for idx in range(PYRAMID_OPEN[lvl]):
        _PYRAMID_SLOTS.append((lvl, idx))
_N_PYRAMID = len(_PYRAMID_SLOTS)  # 12
_PYRAMID_TO_IDX: dict[tuple[int, int], int] = {s: i for i, s in enumerate(_PYRAMID_SLOTS)}

_N_RESERVE_SLOTS = _N_PYRAMID  # 12 (same pyramid slots)
_N_RESERVE_DECK = 3  # L1, L2, L3
_N_BUY_RESERVE = MAX_RESERVED  # 3

# Board cells for scroll / effect
_N_CELLS = BOARD_SIZE * BOARD_SIZE  # 25

_MAX_WILDCARD_TARGETS = 20
_N_ROYAL = 4
_N_OPP_GEM = N_GEMS - 1  # 6 (exclude gold)

# ── Layout: contiguous ranges ─────────────────────────────────────────────────
# [0          .. 145)  TakeTokens
# [145        .. 157)  BuyCard pyramid (12)
# [157        .. 160)  BuyCard reserve (3)
# [160        .. 172)  ReserveCard pyramid (12)
# [172        .. 175)  ReserveCard deck (3)
# [175        .. 200)  UseScroll (25)
# [200        .. 225)  EffectTakeSameGem (25)
# [225        .. 231)  EffectTakeOpponentGem (6)
# [231        .. 251)  EffectChooseWildcard (20)
# [251        .. 255)  ChooseRoyal (4)
# [255        .. 262)  DiscardToken (7)
# [262]               RefillBoard
# [263]               ProceedToMain
# [264]               EffectSkip

OFF_TAKE = 0
OFF_BUY_PYR = OFF_TAKE + _N_TAKE
OFF_BUY_RES = OFF_BUY_PYR + _N_PYRAMID
OFF_RES_PYR = OFF_BUY_RES + _N_BUY_RESERVE
OFF_RES_DECK = OFF_RES_PYR + _N_RESERVE_SLOTS
OFF_SCROLL = OFF_RES_DECK + _N_RESERVE_DECK
OFF_EFFECT_SAME = OFF_SCROLL + _N_CELLS
OFF_EFFECT_OPP = OFF_EFFECT_SAME + _N_CELLS
OFF_EFFECT_WC = OFF_EFFECT_OPP + _N_OPP_GEM
OFF_ROYAL = OFF_EFFECT_WC + _MAX_WILDCARD_TARGETS
OFF_DISCARD = OFF_ROYAL + _N_ROYAL
OFF_REFILL = OFF_DISCARD + N_GEMS
OFF_PROCEED = OFF_REFILL + 1
OFF_SKIP = OFF_PROCEED + 1
N_ACTIONS = OFF_SKIP + 1  # 265


def _cell_idx(r: int, c: int) -> int:
    return r * BOARD_SIZE + c


def _gem_to_opp_idx(gem: int) -> int:
    """Map gem index (0-6, excluding GOLD=6) to 0-5."""
    # gems 0..5 map directly; gold (6) is excluded
    assert gem != Gem.GOLD
    return gem


# ── Public API ────────────────────────────────────────────────────────────────

def action_to_index(action: Action) -> int:
    """Convert a game Action to a fixed action index."""
    if isinstance(action, TakeTokens):
        return OFF_TAKE + _SEG_TO_IDX[action.positions]

    if isinstance(action, BuyCard):
        if action.source == 'pyramid':
            return OFF_BUY_PYR + _PYRAMID_TO_IDX[(action.level, action.index)]
        return OFF_BUY_RES + action.index

    if isinstance(action, ReserveCard):
        if action.source == 'pyramid':
            return OFF_RES_PYR + _PYRAMID_TO_IDX[(action.level, action.index)]
        return OFF_RES_DECK + (action.level - 1)

    if isinstance(action, UseScroll):
        return OFF_SCROLL + _cell_idx(*action.position)

    if isinstance(action, EffectTakeSameGem):
        return OFF_EFFECT_SAME + _cell_idx(*action.position)

    if isinstance(action, EffectTakeOpponentGem):
        return OFF_EFFECT_OPP + _gem_to_opp_idx(action.gem)

    if isinstance(action, EffectChooseWildcard):
        return OFF_EFFECT_WC + action.target_card_index

    if isinstance(action, ChooseRoyal):
        return OFF_ROYAL + action.index

    if isinstance(action, DiscardToken):
        return OFF_DISCARD + action.gem

    if isinstance(action, RefillBoard):
        return OFF_REFILL

    if isinstance(action, ProceedToMain):
        return OFF_PROCEED

    if isinstance(action, EffectSkip):
        return OFF_SKIP

    raise ValueError(f"Unknown action type: {type(action)}")


def index_to_action(index: int) -> Action:
    """
    Convert a fixed action index back to a game Action.

    Note: the returned Action is structurally valid but may not be legal
    in a given state. Always use with legal_mask.
    """
    if OFF_TAKE <= index < OFF_BUY_PYR:
        seg = ALL_SEGMENTS[index - OFF_TAKE]
        return TakeTokens(positions=seg)

    if OFF_BUY_PYR <= index < OFF_BUY_RES:
        lvl, idx = _PYRAMID_SLOTS[index - OFF_BUY_PYR]
        return BuyCard(source='pyramid', level=lvl, index=idx)

    if OFF_BUY_RES <= index < OFF_RES_PYR:
        return BuyCard(source='reserve', level=0, index=index - OFF_BUY_RES)

    if OFF_RES_PYR <= index < OFF_RES_DECK:
        lvl, idx = _PYRAMID_SLOTS[index - OFF_RES_PYR]
        return ReserveCard(source='pyramid', level=lvl, index=idx)

    if OFF_RES_DECK <= index < OFF_SCROLL:
        lvl = (index - OFF_RES_DECK) + 1
        return ReserveCard(source='deck', level=lvl, index=0)

    if OFF_SCROLL <= index < OFF_EFFECT_SAME:
        ci = index - OFF_SCROLL
        return UseScroll(position=(ci // BOARD_SIZE, ci % BOARD_SIZE))

    if OFF_EFFECT_SAME <= index < OFF_EFFECT_OPP:
        ci = index - OFF_EFFECT_SAME
        return EffectTakeSameGem(position=(ci // BOARD_SIZE, ci % BOARD_SIZE))

    if OFF_EFFECT_OPP <= index < OFF_EFFECT_WC:
        gem = index - OFF_EFFECT_OPP
        return EffectTakeOpponentGem(gem=gem)

    if OFF_EFFECT_WC <= index < OFF_ROYAL:
        return EffectChooseWildcard(target_card_index=index - OFF_EFFECT_WC)

    if OFF_ROYAL <= index < OFF_DISCARD:
        return ChooseRoyal(index=index - OFF_ROYAL)

    if OFF_DISCARD <= index < OFF_REFILL:
        return DiscardToken(gem=index - OFF_DISCARD)

    if index == OFF_REFILL:
        return RefillBoard()

    if index == OFF_PROCEED:
        return ProceedToMain()

    if index == OFF_SKIP:
        return EffectSkip()

    raise ValueError(f"Invalid action index: {index}")


def legal_mask(state: GameState) -> np.ndarray:
    """
    Return a boolean mask of length N_ACTIONS.
    mask[i] == True iff action i is legal in this state.
    """
    actions = GameEngine.get_legal_actions(state)
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    for a in actions:
        mask[action_to_index(a)] = True
    return mask


def legal_actions_with_indices(state: GameState) -> list[tuple[int, Action]]:
    """Return list of (index, Action) for all legal actions."""
    actions = GameEngine.get_legal_actions(state)
    return [(action_to_index(a), a) for a in actions]
