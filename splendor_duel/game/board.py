"""
board.py — 5×5 game board backed by a numpy int8 array.

Encoding:
  -1  →  empty cell  (EMPTY constant)
   0  →  Gem.WHITE
   1  →  Gem.BLACK
   ...
   6  →  Gem.GOLD

All mutating methods return a NEW Board instance (immutable design for MCTS).
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .constants import (
    ALL_POSITIONS,
    BOARD_SIZE,
    BOARD_SPIRAL,
    EMPTY,
    EMPTY_INT,
    GOLD_INT,
    SEGMENT_MASKS,
    TOKEN_COUNTS,
    Gem,
    GEM_NAMES,
    N_GEMS,
)


class Board:
    """
    Immutable-style 5×5 token board.

    The internal numpy array is int8.  Callers treat Board instances as
    read-only; every operation that changes state returns a new Board.
    """

    __slots__ = ('_grid',)

    def __init__(self, grid: np.ndarray) -> None:
        assert grid.shape == (BOARD_SIZE, BOARD_SIZE)
        assert grid.dtype == np.int8
        self._grid = grid

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def empty(cls) -> Board:
        """Create a board with all cells empty."""
        return cls(np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=np.int8))

    @classmethod
    def initial(cls) -> tuple[Board, dict[str, int]]:
        """
        Create a randomly filled board and return (board, remaining_bag).

        Follows the setup rule: place all 25 tokens randomly, starting
        from the central cell and following the spiral.  The remaining_bag
        is always empty after initial setup because TOKEN_COUNTS sums to 25.
        """
        tokens = []
        for name, count in TOKEN_COUNTS.items():
            gem_idx = GEM_NAMES.index(name)
            tokens.extend([gem_idx] * count)
        random.shuffle(tokens)

        grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=np.int8)
        for (r, c), gem_idx in zip(BOARD_SPIRAL, tokens):
            grid[r, c] = gem_idx

        bag: dict[str, int] = {name: 0 for name in GEM_NAMES}
        return cls(grid), bag

    def copy(self) -> Board:
        return Board(self._grid.copy())

    # ── Read-only access ──────────────────────────────────────────────────────

    @property
    def grid(self) -> np.ndarray:
        """Read-only view of the underlying array."""
        view = self._grid.view()
        view.flags.writeable = False
        return view

    def __getitem__(self, pos: tuple[int, int]) -> int:
        return int(self._grid[pos])

    def is_empty(self, r: int, c: int) -> bool:
        return self._grid[r, c] == EMPTY

    def token_at(self, r: int, c: int) -> Optional[int]:
        """Return Gem index or None if cell is empty."""
        v = int(self._grid[r, c])
        return None if v == EMPTY else v

    def count_tokens(self) -> dict[str, int]:
        """Count remaining tokens on the board by gem name."""
        counts = {name: 0 for name in GEM_NAMES}
        for idx in range(N_GEMS):
            counts[GEM_NAMES[idx]] = int(np.sum(self._grid == idx))
        return counts

    def n_empty(self) -> int:
        return int(np.sum(self._grid == EMPTY))

    # ── Legal action generation ───────────────────────────────────────────────

    def get_legal_take_positions(self) -> list[tuple[tuple[int, int], ...]]:
        """
        Return all legal TakeTokens selections.

        Rules:
        - 1, 2, or 3 tokens in a contiguous horizontal / vertical / diagonal line
        - All selected cells must be non-empty and non-gold
        - Special privilege trigger (3 of same colour, 2 pearls) is noted
          in apply_take but NOT filtered here — all valid geometries are returned.

        Builds a 25-bit mask of unusable (empty or gold) cells, then keeps every
        pre-computed segment mask (SEGMENT_MASKS) that shares no bit with it.
        `.tolist()` is the only numpy call: it unboxes all 25 cells to native
        ints in one shot, so the per-cell loop never re-enters numpy.
        """
        blocked = 0
        for i, v in enumerate(self._grid.reshape(-1).tolist()):
            if v == EMPTY_INT or v == GOLD_INT:
                blocked |= 1 << i

        return [seg for seg, mask in SEGMENT_MASKS if not (mask & blocked)]

    def get_legal_single_positions(
        self, exclude_gold: bool = True
    ) -> list[tuple[int, int]]:
        """
        Return all cells with a token (optionally excluding gold).
        Used for scroll-spend token selection.
        """
        cells = self._grid.reshape(-1).tolist()
        if exclude_gold:
            return [
                pos for pos, v in zip(ALL_POSITIONS, cells)
                if v != EMPTY_INT and v != GOLD_INT
            ]
        return [pos for pos, v in zip(ALL_POSITIONS, cells) if v != EMPTY_INT]

    def has_gold(self) -> bool:
        return bool(np.any(self._grid == Gem.GOLD))

    # ── State transitions (return new Board) ──────────────────────────────────

    def take_tokens(
        self, positions: tuple[tuple[int, int], ...]
    ) -> tuple[Board, np.ndarray]:
        """
        Remove tokens at the given positions.

        Returns:
            new_board   — board with those cells set to EMPTY
            taken_vec   — int8 numpy vector (length N_GEMS) of taken counts
        """
        new_grid = self._grid.copy()
        taken_vec = np.zeros(N_GEMS, dtype=np.int8)

        for r, c in positions:
            gem = int(new_grid[r, c])
            assert gem != EMPTY, f"Cell ({r},{c}) is already empty"
            taken_vec[gem] += 1
            new_grid[r, c] = EMPTY

        return Board(new_grid), taken_vec

    def take_single(self, r: int, c: int) -> tuple[Board, int]:
        """
        Remove a single token (for scroll-spend or take_same_gem effect).

        Returns (new_board, gem_index).
        """
        gem = int(self._grid[r, c])
        assert gem != EMPTY, f"Cell ({r},{c}) is empty"
        new_grid = self._grid.copy()
        new_grid[r, c] = EMPTY
        return Board(new_grid), gem

    def refill(self, bag: dict[str, int]) -> tuple[Board, dict[str, int], bool]:
        """
        Fill empty cells from the bag following BOARD_SPIRAL order.

        Tokens in the bag are shuffled before placement (per rules: 'mix
        tokens in bag, then place').

        Returns:
            new_board     — board with newly placed tokens
            remaining_bag — updated bag counts
            did_refill    — False if bag was already empty (action is illegal)
        """
        total_in_bag = sum(bag.values())
        if total_in_bag == 0:
            return self, bag, False

        # Build a flat list of gem indices available in the bag, shuffle it
        available = []
        for name, count in bag.items():
            available.extend([GEM_NAMES.index(name)] * count)
        random.shuffle(available)

        new_grid = self._grid.copy()
        new_bag  = bag.copy()
        placed   = 0

        for r, c in BOARD_SPIRAL:
            if new_grid[r, c] != EMPTY:
                continue                   # already occupied
            if placed >= len(available):
                break                      # bag exhausted
            gem_idx = available[placed]
            new_grid[r, c] = gem_idx
            new_bag[GEM_NAMES[gem_idx]] -= 1
            placed += 1

        return Board(new_grid), new_bag, True

    # ── Trigger detection ─────────────────────────────────────────────────────

    @staticmethod
    def triggers_privilege(
        positions: tuple[tuple[int, int], ...],
        taken_vec: np.ndarray,
    ) -> bool:
        """
        Return True if this take-tokens action gives the opponent a scroll.

        Triggers:
        - Taking 3 tokens of the same colour
        - Taking 2 pearl tokens
        """
        # 3 of same colour: any single non-pearl, non-gold gem with count 3
        for gem in [Gem.WHITE, Gem.BLACK, Gem.RED, Gem.BLUE, Gem.GREEN]:
            if taken_vec[gem] == 3:
                return True
        # 2 pearls
        if taken_vec[Gem.PEARL] == 2:
            return True
        return False

    # ── Display ───────────────────────────────────────────────────────────────

    SYMBOLS = {
        -1: '·',
        Gem.WHITE: 'W',
        Gem.BLACK: 'K',
        Gem.RED:   'R',
        Gem.BLUE:  'B',
        Gem.GREEN: 'G',
        Gem.PEARL: 'P',
        Gem.GOLD:  '$',
    }

    def __str__(self) -> str:
        rows = []
        for r in range(BOARD_SIZE):
            row = ' '.join(self.SYMBOLS[int(self._grid[r, c])]
                           for c in range(BOARD_SIZE))
            rows.append(row)
        return '\n'.join(rows)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return np.array_equal(self._grid, other._grid)

    def __repr__(self) -> str:
        return f"Board(\n{self}\n)"