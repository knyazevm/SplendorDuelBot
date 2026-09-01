"""
observation_v2.py — Extended observation: the state the 519-dim encoding omits.

Three pieces of decision-relevant state are simply absent from
`observation.encode_state`, and the network has no way to infer them:

  1. THE OPPONENT'S RESERVED CARDS. `encode_state` writes "My reserved" (3x17)
     but for the opponent stores only a COUNT. Up to three cards that determine
     what the opponent can buy next are invisible. In Splendor Duel, reserving
     from the pyramid is public — both players watch it happen — so this is not
     hidden information the agent is meant to lack; it is information the
     encoding drops. (MCTS could see it all along via GameState, which also
     made its policy targets partly unlearnable.)

  2. `pending_effect` / `pending_card`. In the EFFECT phase the network sees
     only a "phase=EFFECT" one-hot. Which ability is being resolved
     (take_same_gem / take_scroll / take_opponent_gem / extra_turn) and which
     card triggered it are not encoded, yet they determine the legal actions
     and the right choice entirely.

  3. `extra_turn_flag` — whether the player is about to act again.

Layout: the original 519 dims verbatim, then 74 appended, so a v1 network's
learned features map onto the same input positions.

    opponent reserved   3 x 17 = 51
    pending_effect one-hot     =  5
    pending_card               = 17
    extra_turn_flag            =  1
    ------------------------------- 74      OBS_SIZE_V2 = 593
"""
from __future__ import annotations

import numpy as np

from splendor_duel.game.constants import MAX_RESERVED
from splendor_duel.game.state import GameState

from .observation import (
    OBS_SIZE, _ABILITY_IDS, _CARD_DIM, _encode_card, encode_state,
)

_N_OPP_RESERVED = MAX_RESERVED * _CARD_DIM   # 51
_N_PENDING_EFFECT = 5
_N_PENDING_CARD = _CARD_DIM                  # 17
_N_EXTRA_TURN = 1

_N_EXTRA = _N_OPP_RESERVED + _N_PENDING_EFFECT + _N_PENDING_CARD + _N_EXTRA_TURN  # 74
OBS_SIZE_V2 = OBS_SIZE + _N_EXTRA                                                 # 593


def encode_state_v2(state: GameState) -> np.ndarray:
    """Encode as v1 (first 519 dims), then append the omitted state."""
    obs = np.zeros(OBS_SIZE_V2, dtype=np.float32)
    obs[:OBS_SIZE] = encode_state(state)
    off = OBS_SIZE

    # Opponent's reserved cards — same encoding as "my reserved".
    opp = state.opponent
    for i in range(MAX_RESERVED):
        if i < len(opp.reserved):
            off = _encode_card(obs, off, opp.reserved[i])
        else:
            off += _CARD_DIM

    # Which ability is being resolved right now.
    obs[off + _ABILITY_IDS.get(state.pending_effect, 0)] = 1.0
    off += _N_PENDING_EFFECT

    # The card that triggered it.
    if state.pending_card is not None:
        off = _encode_card(obs, off, state.pending_card)
    else:
        off += _N_PENDING_CARD

    obs[off] = 1.0 if state.extra_turn_flag else 0.0
    off += _N_EXTRA_TURN

    assert off == OBS_SIZE_V2, f"{off} != {OBS_SIZE_V2}"
    return obs
