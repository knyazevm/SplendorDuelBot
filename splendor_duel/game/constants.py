from enum import IntEnum
import numpy as np


class Gem(IntEnum):
    WHITE = 0
    BLACK = 1
    RED = 2
    BLUE = 3
    GREEN = 4
    PEARL = 5
    GOLD = 6


GEM_NAMES = ['white', 'black', 'red', 'blue', 'green', 'pearl', 'gold']
N_GEMS = len(GEM_NAMES)  # 7
BOARD_SIZE = 5

# ── Token supply in the bag ───────────────────────────────────────────────────
TOKEN_COUNTS: dict[str, int] = {
    'white': 4, 'black': 4, 'red': 4, 'blue': 4, 'green': 4,
    'pearl': 2, 'gold': 3,
}  # total = 25, fills the 5×5 board exactly

# ── Board encoding ────────────────────────────────────────────────────────────
EMPTY: np.int8 = np.int8(-1)  # empty cell marker in the numpy board

# ── Spiral fill order (row, col), starting from center, clockwise ─────────────
# Used when refilling the board from the bag.
BOARD_SPIRAL: list[tuple[int, int]] = [
    (2, 2),  # center
    (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1),  # ring 1
    (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),  # ring 2 …
    (4, 3), (4, 2), (4, 1), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1),
]
assert len(BOARD_SPIRAL) == BOARD_SIZE ** 2 == 25


# ── Precompute ALL candidate line segments on the board ───────────────────────
# A segment is a tuple of (row, col) positions forming a contiguous line
# in one of four directions, with length 1–3.
# Used by Board.get_legal_take_positions().
def _build_segments() -> list[tuple[tuple[int, int], ...]]:
    seen: set = set()
    segments: list = []
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for length in range(1, 4):
                    seg = tuple(
                        (r + i * dr, c + i * dc)
                        for i in range(length)
                    )
                    if (
                            all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE
                                for rr, cc in seg)
                            and seg not in seen
                    ):
                        seen.add(seg)
                        segments.append(seg)
    return segments


ALL_SEGMENTS: list[tuple[tuple[int, int], ...]] = _build_segments()

# ── Pyramid layout ────────────────────────────────────────────────────────────
PYRAMID_OPEN: dict[int, int] = {1: 5, 2: 4, 3: 3}  # visible cards per level

# ── Game limits ───────────────────────────────────────────────────────────────
MAX_TOKENS = 10
MAX_RESERVED = 3
MAX_SCROLLS = 3
N_ROYAL_CARDS = 4

CROWNS_ROYAL_1 = 3  # take first royal card
CROWNS_ROYAL_2 = 6  # take second royal card

# ── Victory thresholds ────────────────────────────────────────────────────────
VP_WIN = 20  # total prestige points
CROWNS_WIN = 10  # total crowns
MONO_VP_WIN = 10  # prestige points in cards of a single bonus colour

# ── Card ability identifiers (must match cards.json) ─────────────────────────
ABILITY_EXTRA_TURN = 'extra_turn'
ABILITY_TAKE_SAME_GEM = 'take_same_gem'
ABILITY_TAKE_SCROLL = 'take_scroll'
ABILITY_TAKE_OPPONENT_GEM = 'take_opponent_gem'
