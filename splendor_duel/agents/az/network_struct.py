"""
network_struct.py — Policy/value network with a STRUCTURED policy head.

STATUS: MEASURED, NO BENEFIT. Kept as a recorded negative result — do not
reach for this again without new evidence. A/B against the flat head on an
85/15 whole-game split of the v2 replay buffer (both 256x256x256 trunk, AdamW
1e-3, identical batches, 1200 steps) gave essentially identical held-out
policy cross-entropy:

    MAIN-phase excess over target entropy:  flat +0.971   structured +0.981

The reason the flat head was suspected turned out to be a measurement error on
my part. "Excess over target entropy" is not a model-capacity gap, because the
target entropy is not an achievable floor: the target is one NOISY SAMPLE from
MCTS, so the best possible predictor still pays H(target) + sampling noise.
Measured directly — two independent searches from the same MAIN position, same
net, same sim budget — the two targets disagree by 2.969 nats, while the
network's excess is only 0.97. The policy head already predicts the target
distribution three times better than one search predicts another.

The real constraint is target NOISE: ~234 sims spread over ~48 legal MAIN
actions is ~5 visits per action. The fix is more simulations per move (i.e.
self-play throughput), not a different head.

Motivation (measured, see scripts/train_az_v2.py header for the run):
the flat MLP head plateaus at held-out policy CE ~2.25 against a target
entropy of ~1.62, and the gap does not close with training — held-out CE
*rises* to 2.38 while train CE falls to 1.97. Broken down by phase, essentially
all of that excess sits in MAIN (1.009 nats over 5796 positions).

MAIN is where the take-token actions live, and they are 145 of the 278 action
slots (55%). Each one corresponds to a fixed geometric line segment of the 5x5
board (`constants.ALL_SEGMENTS`: 25 of length 1, 72 of length 2, 48 of length
3). A flat `Linear(hidden, 278)` has to learn, separately for each of those 145
slots, which cells of a 200-dim flattened one-hot board it should read. Nothing
ties slot 37 to the three cells it actually covers, and nothing lets "take three
red" generalise from one part of the board to another.

This head instead COMPUTES each take logit from the cells that segment covers:

    cell_emb  = CellEncoder(board)          [B, 25, d]
    seg_emb   = INCIDENCE @ cell_emb        [B, 145, d]   (sum over the
                                                           segment's own cells)
    take_logit= TakeHead([seg_emb, ctx])    [B, 145]

`INCIDENCE` is a fixed (non-learned) [145, 25] 0/1 matrix built once from
ALL_SEGMENTS, so the pooling is a single matmul. The same treatment is applied
to the 24 pyramid card slots (buy-pyramid and reserve-pyramid), whose logits are
computed from that slot's own 17-dim card encoding rather than from its index.
The remaining 96 slots (scroll, effects, royal, discard, refill/proceed/skip)
keep a flat head — they are few, and not positionally structured in the same way.

Contracts preserved: OBS_SIZE stays 519 and every action-space offset is read
from action_map rather than hardcoded, so action_map,
legal_mask, mcts_az, self_play_v2 and the trainer are all unchanged. Only the
parameters differ, so this does NOT load older checkpoints.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from splendor_duel.env import N_ACTIONS, OBS_SIZE
from splendor_duel.env.action_map import (
    OFF_BUY_PYR, OFF_BUY_RES, OFF_RES_PYR, OFF_RES_DECK, OFF_TAKE,
)
from splendor_duel.game.constants import ALL_SEGMENTS, BOARD_SIZE

# ── Observation layout (see env/observation.py docstring) ─────────────────────
# Board occupies the first 200 dims as 25 cells x 8 channels, row-major.
_N_CELLS = BOARD_SIZE * BOARD_SIZE          # 25
_CELL_CH = 8                                 # 7 gems + empty
_BOARD_END = _N_CELLS * _CELL_CH             # 200
# 200 board + 23 me + 23 opponent = 246, then 12 pyramid cards x 17.
_PYR_START = _BOARD_END + 23 * 2             # 246
_CARD_DIM = 17
_N_PYR = 12
_PYR_END = _PYR_START + _N_PYR * _CARD_DIM   # 450

_N_TAKE = len(ALL_SEGMENTS)                  # 145


def _build_incidence() -> np.ndarray:
    """[145, 25] 0/1 matrix: which board cells each take-token segment covers."""
    inc = np.zeros((_N_TAKE, _N_CELLS), dtype=np.float32)
    for i, seg in enumerate(ALL_SEGMENTS):
        for (r, c) in seg:
            inc[i, r * BOARD_SIZE + c] = 1.0
    return inc


class AZStructuredNetwork(nn.Module):
    """Shared trunk + structured policy head + tanh value head."""

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256, 256, 256),
        cell_dim: int = 24,
        card_dim: int = 24,
        ctx_dim: int = 48,
        head_hidden: int = 48,
    ):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.cell_dim = cell_dim
        self.card_dim = card_dim

        layers: list[nn.Module] = []
        in_size = OBS_SIZE
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.LayerNorm(h), nn.ReLU()]
            in_size = h
        self.shared = nn.Sequential(*layers)
        self.trunk_out = in_size

        # Fixed geometry: segment -> cells. Registered as a buffer so it moves
        # with .to(device) and is saved, but never receives gradient.
        inc = _build_incidence()
        self.register_buffer("incidence", torch.from_numpy(inc))
        # Segment length (1/2/3) one-hot — "take 3" and "take 1" differ in kind.
        seg_len = inc.sum(1).astype(np.int64) - 1
        self.register_buffer(
            "seg_len_onehot", torch.eye(3)[torch.from_numpy(seg_len)]
        )

        self.cell_enc = nn.Sequential(
            nn.Linear(_CELL_CH, cell_dim), nn.LayerNorm(cell_dim), nn.ReLU(),
        )
        # Lets the head express board-position preferences the pooled content
        # cannot (edges vs centre), without reintroducing per-slot memorisation.
        self.cell_pos = nn.Parameter(torch.zeros(_N_CELLS, cell_dim))

        self.ctx_proj = nn.Sequential(nn.Linear(self.trunk_out, ctx_dim), nn.ReLU())

        self.take_head = nn.Sequential(
            nn.Linear(cell_dim + 3 + ctx_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self.card_enc = nn.Sequential(
            nn.Linear(_CARD_DIM, card_dim), nn.LayerNorm(card_dim), nn.ReLU(),
        )
        self.buy_head = nn.Sequential(
            nn.Linear(card_dim + ctx_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        self.reserve_head = nn.Sequential(
            nn.Linear(card_dim + ctx_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Covers the slots with no positional structure worth exploiting.
        self.flat_head = nn.Linear(self.trunk_out, N_ACTIONS)
        self.value_head = nn.Linear(self.trunk_out, 1)

        nn.init.orthogonal_(self.flat_head.weight, gain=0.01)
        nn.init.zeros_(self.flat_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        for head in (self.take_head, self.buy_head, self.reserve_head):
            nn.init.orthogonal_(head[-1].weight, gain=0.01)
            nn.init.zeros_(head[-1].bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        feat = self.shared(obs)
        ctx = self.ctx_proj(feat)                                  # [B, ctx]
        flat = self.flat_head(feat)                                # [B, N_ACTIONS]

        # ── Take-token logits from the cells each segment covers ──
        board = obs[:, :_BOARD_END].view(B, _N_CELLS, _CELL_CH)
        cell = self.cell_enc(board) + self.cell_pos                # [B, 25, d]
        seg = torch.matmul(self.incidence, cell)                   # [B, 145, d]
        seg = torch.cat([
            seg,
            self.seg_len_onehot.expand(B, -1, -1),
            ctx.unsqueeze(1).expand(-1, _N_TAKE, -1),
        ], dim=-1)
        take = self.take_head(seg).squeeze(-1)                     # [B, 145]

        # ── Pyramid buy / reserve logits from that slot's own card ──
        cards = obs[:, _PYR_START:_PYR_END].view(B, _N_PYR, _CARD_DIM)
        card = self.card_enc(cards)                                # [B, 12, d]
        card_ctx = torch.cat(
            [card, ctx.unsqueeze(1).expand(-1, _N_PYR, -1)], dim=-1)
        buy = self.buy_head(card_ctx).squeeze(-1)                  # [B, 12]
        reserve = self.reserve_head(card_ctx).squeeze(-1)          # [B, 12]

        # Reassemble in action_map order; slices between structured blocks come
        # from the flat head so every one of the N_ACTIONS slots is covered exactly once.
        logits = torch.cat([
            take,                              # [0, 145)   TakeTokens
            buy,                               # [145, 157) BuyCard pyramid
            flat[:, OFF_BUY_RES:OFF_RES_PYR],  # [157, 160) BuyCard reserve
            reserve,                           # [160, 172) ReserveCard pyramid
            flat[:, OFF_RES_DECK:],            # [172, N_ACTIONS) everything else
        ], dim=1).clamp(-15, 15)

        return logits, torch.tanh(self.value_head(feat))


def _assert_layout():
    """Fail loudly if the action-space or observation layout ever shifts."""
    assert OFF_TAKE == 0 and OFF_BUY_PYR == _N_TAKE, "take/buy offsets moved"
    assert OFF_BUY_RES == OFF_BUY_PYR + _N_PYR, "buy-reserve offset moved"
    assert OFF_RES_PYR == OFF_BUY_RES + 3, "reserve-pyramid offset moved"
    assert OFF_RES_DECK == OFF_RES_PYR + _N_PYR, "reserve-deck offset moved"
    assert _PYR_END <= OBS_SIZE, "pyramid block runs past the observation"


_assert_layout()
