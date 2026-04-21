"""
Tests for constants.py, card.py, board.py.
Run with:  pytest tests/test_board.py -v
"""
import json
import os
import tempfile

import numpy as np
import pytest

from splendor_duel.game.constants import (
    ALL_SEGMENTS,
    BOARD_SPIRAL,
    BOARD_SIZE,
    EMPTY,
    TOKEN_COUNTS,
    Gem,
    GEM_NAMES,
    N_GEMS,
)
from splendor_duel.game.card import Card, RoyalCard, load_cards
from splendor_duel.game.board import Board


# ══════════════════════════════════════════════════════════════════════════════
# constants.py
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:

    def test_gem_count(self):
        assert N_GEMS == 7

    def test_gem_names_order(self):
        assert GEM_NAMES[Gem.WHITE] == 'white'
        assert GEM_NAMES[Gem.GOLD]  == 'gold'
        assert GEM_NAMES[Gem.PEARL] == 'pearl'

    def test_token_counts_sum_to_25(self):
        assert sum(TOKEN_COUNTS.values()) == BOARD_SIZE ** 2 == 25

    def test_spiral_length(self):
        assert len(BOARD_SPIRAL) == 25

    def test_spiral_starts_at_center(self):
        assert BOARD_SPIRAL[0] == (2, 2)

    def test_spiral_covers_all_cells(self):
        cells = set(BOARD_SPIRAL)
        expected = {(r, c) for r in range(5) for c in range(5)}
        assert cells == expected

    def test_all_segments_no_duplicates(self):
        seen = set()
        for seg in ALL_SEGMENTS:
            assert seg not in seen, f"Duplicate segment: {seg}"
            seen.add(seg)

    def test_all_segments_in_bounds(self):
        for seg in ALL_SEGMENTS:
            for r, c in seg:
                assert 0 <= r < BOARD_SIZE
                assert 0 <= c < BOARD_SIZE

    def test_all_segments_length_1_to_3(self):
        for seg in ALL_SEGMENTS:
            assert 1 <= len(seg) <= 3

    def test_all_segments_are_collinear(self):
        """Every segment of length >= 2 must form a straight line."""
        for seg in ALL_SEGMENTS:
            if len(seg) < 2:
                continue
            dr = seg[1][0] - seg[0][0]
            dc = seg[1][1] - seg[0][1]
            for i in range(1, len(seg)):
                assert seg[i][0] - seg[i-1][0] == dr
                assert seg[i][1] - seg[i-1][1] == dc

    def test_all_single_cell_segments_present(self):
        singles = {seg[0] for seg in ALL_SEGMENTS if len(seg) == 1}
        expected = {(r, c) for r in range(5) for c in range(5)}
        assert singles == expected


# ══════════════════════════════════════════════════════════════════════════════
# card.py
# ══════════════════════════════════════════════════════════════════════════════

MINIMAL_JSON = {
    "cards": [
        {
            "id": "L1_01",
            "level": 1,
            "cost": {"white": 2, "black": 0, "red": 1,
                     "blue": 0, "green": 0, "pearl": 0, "gold": 0},
            "gem_bonus": {"green": 1},
            "points": 0,
            "crowns": 1,
            "ability": None,
        },
        {
            "id": "L2_01",
            "level": 2,
            "cost": {"white": 3, "black": 0, "red": 0,
                     "blue": 2, "green": 0, "pearl": 1, "gold": 0},
            "gem_bonus": {"wildcard": 1},
            "points": 2,
            "crowns": 0,
            "ability": "extra_turn",
        },
        {
            "id": "L3_01",
            "level": 3,
            "cost": {"white": 0, "black": 5, "red": 0,
                     "blue": 0, "green": 3, "pearl": 0, "gold": 0},
            "gem_bonus": {"black": 2},
            "points": 4,
            "crowns": 0,
            "ability": None,
        },
    ],
    "royal_cards": [
        {"id": "R_01", "points": 2, "ability": "take_scroll"},
        {"id": "R_02", "points": 3, "ability": None},
    ],
}


@pytest.fixture
def cards_json(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(MINIMAL_JSON))
    return str(path)


class TestCard:

    def test_load_cards_count(self, cards_json):
        cards, royals = load_cards(cards_json)
        assert len(cards) == 3
        assert len(royals) == 2

    def test_regular_card_fields(self, cards_json):
        cards, _ = load_cards(cards_json)
        c = cards[0]   # L1_01
        assert c.id == 'L1_01'
        assert c.level == 1
        assert c.crowns == 1
        assert c.points == 0
        assert not c.is_wildcard
        assert c.ability is None

    def test_cost_vector_length(self, cards_json):
        cards, _ = load_cards(cards_json)
        for card in cards:
            assert card.cost_vec.shape == (N_GEMS,)

    def test_cost_vector_values(self, cards_json):
        cards, _ = load_cards(cards_json)
        c = cards[0]  # white=2, red=1, rest=0
        assert c.cost_vec[Gem.WHITE] == 2
        assert c.cost_vec[Gem.RED]   == 1
        assert c.cost_vec[Gem.BLUE]  == 0

    def test_bonus_vector_green(self, cards_json):
        cards, _ = load_cards(cards_json)
        c = cards[0]  # gem_bonus: green 1
        assert c.bonus_vec[Gem.GREEN] == 1
        assert c.bonus_vec.sum() == 1

    def test_wildcard_card(self, cards_json):
        cards, _ = load_cards(cards_json)
        c = cards[1]  # L2_01, wildcard
        assert c.is_wildcard
        assert c.gem_bonus is None
        assert np.all(c.bonus_vec == 0)   # wildcard has no fixed bonus

    def test_double_bonus(self, cards_json):
        cards, _ = load_cards(cards_json)
        c = cards[2]  # L3_01, black: 2
        assert c.bonus_vec[Gem.BLACK] == 2

    def test_royal_card_fields(self, cards_json):
        _, royals = load_cards(cards_json)
        r0, r1 = royals
        assert r0.id == 'R_01'
        assert r0.points == 2
        assert r0.ability == 'take_scroll'
        assert r1.points == 3
        assert r1.ability is None

    def test_card_is_hashable(self, cards_json):
        cards, _ = load_cards(cards_json)
        s = set(cards)
        assert len(s) == 3


# ══════════════════════════════════════════════════════════════════════════════
# board.py
# ══════════════════════════════════════════════════════════════════════════════

class TestBoardConstructors:

    def test_empty_board(self):
        b = Board.empty()
        assert np.all(b.grid == EMPTY)
        assert b.n_empty() == 25

    def test_initial_board_fills_all_cells(self):
        b, bag = Board.initial()
        assert b.n_empty() == 0
        assert sum(bag.values()) == 0  # bag is fully placed

    def test_initial_board_token_counts(self):
        b, _ = Board.initial()
        counts = b.count_tokens()
        for name, expected in TOKEN_COUNTS.items():
            assert counts[name] == expected, f"Mismatch for {name}"

    def test_copy_is_independent(self):
        b, _ = Board.initial()
        b2 = b.copy()
        assert b == b2
        # Mutate original (internal) — copy must not change
        b._grid[0, 0] = EMPTY
        assert b != b2


class TestBoardLegalLines:

    def _board_with(self, mapping: dict[tuple, int]) -> Board:
        """Create a board with specific gems at given (r,c) positions."""
        grid = np.full((5, 5), EMPTY, dtype=np.int8)
        for (r, c), gem in mapping.items():
            grid[r, c] = gem
        return Board(grid)

    def test_empty_board_no_legal_lines(self):
        b = Board.empty()
        assert b.get_legal_take_positions() == []

    def test_single_token_one_legal(self):
        b = self._board_with({(2, 2): Gem.RED})
        legal = b.get_legal_take_positions()
        assert ((2, 2),) in legal

    def test_gold_not_takeable(self):
        b = self._board_with({(2, 2): Gem.GOLD})
        legal = b.get_legal_take_positions()
        assert legal == []

    def test_gold_breaks_line(self):
        # R G R horizontally — gold in middle breaks the 3-token line
        b = self._board_with({
            (2, 0): Gem.RED,
            (2, 1): Gem.GOLD,
            (2, 2): Gem.RED,
        })
        legal = b.get_legal_take_positions()
        # The full 3-token segment (2,0)-(2,1)-(2,2) must NOT be legal
        assert ((2, 0), (2, 1), (2, 2)) not in legal
        # But single reds are legal
        assert ((2, 0),) in legal
        assert ((2, 2),) in legal

    def test_horizontal_3_token_line(self):
        b = self._board_with({
            (1, 1): Gem.WHITE,
            (1, 2): Gem.BLACK,
            (1, 3): Gem.GREEN,
        })
        legal = b.get_legal_take_positions()
        assert ((1, 1), (1, 2), (1, 3)) in legal

    def test_diagonal_line(self):
        b = self._board_with({
            (0, 0): Gem.BLUE,
            (1, 1): Gem.RED,
            (2, 2): Gem.GREEN,
        })
        legal = b.get_legal_take_positions()
        assert ((0, 0), (1, 1), (2, 2)) in legal

    def test_empty_cell_breaks_line(self):
        b = self._board_with({
            (0, 0): Gem.BLUE,
            # (1,1) empty
            (2, 2): Gem.GREEN,
        })
        legal = b.get_legal_take_positions()
        assert ((0, 0), (1, 1), (2, 2)) not in legal


class TestBoardTakeTokens:

    def _simple_board(self) -> Board:
        grid = np.full((5, 5), EMPTY, dtype=np.int8)
        grid[2, 2] = Gem.RED
        grid[2, 3] = Gem.BLUE
        grid[2, 4] = Gem.GREEN
        return Board(grid)

    def test_take_single_token(self):
        b = self._simple_board()
        new_b, taken = b.take_tokens(((2, 2),))
        assert new_b.is_empty(2, 2)
        assert taken[Gem.RED] == 1
        assert taken.sum() == 1

    def test_take_three_tokens(self):
        b = self._simple_board()
        new_b, taken = b.take_tokens(((2, 2), (2, 3), (2, 4)))
        assert new_b.is_empty(2, 2)
        assert new_b.is_empty(2, 3)
        assert new_b.is_empty(2, 4)
        assert taken[Gem.RED]   == 1
        assert taken[Gem.BLUE]  == 1
        assert taken[Gem.GREEN] == 1

    def test_original_board_unchanged(self):
        b = self._simple_board()
        _ = b.take_tokens(((2, 2),))
        assert not b.is_empty(2, 2)   # original intact


class TestBoardTriggerPrivilege:

    def test_3_same_colour_triggers(self):
        taken = np.zeros(N_GEMS, dtype=np.int8)
        taken[Gem.RED] = 3
        assert Board.triggers_privilege((), taken) is True

    def test_2_pearls_triggers(self):
        taken = np.zeros(N_GEMS, dtype=np.int8)
        taken[Gem.PEARL] = 2
        assert Board.triggers_privilege((), taken) is True

    def test_3_mixed_no_trigger(self):
        taken = np.zeros(N_GEMS, dtype=np.int8)
        taken[Gem.RED]   = 1
        taken[Gem.BLUE]  = 1
        taken[Gem.GREEN] = 1
        assert Board.triggers_privilege((), taken) is False

    def test_1_pearl_no_trigger(self):
        taken = np.zeros(N_GEMS, dtype=np.int8)
        taken[Gem.PEARL] = 1
        assert Board.triggers_privilege((), taken) is False


class TestBoardRefill:

    def test_refill_empty_bag_fails(self):
        b = Board.empty()
        bag = {name: 0 for name in GEM_NAMES}
        _, _, did_refill = b.refill(bag)
        assert did_refill is False

    def test_refill_places_tokens_in_spiral_order(self):
        b = Board.empty()
        bag = {'red': 3, 'white': 0, 'black': 0,
               'blue': 0, 'green': 0, 'pearl': 0, 'gold': 0}
        new_b, new_bag, did_refill = b.refill(bag)
        assert did_refill is True
        assert sum(new_bag.values()) == 0
        # All 3 red tokens must be on the board
        counts = new_b.count_tokens()
        assert counts['red'] == 3

    def test_refill_respects_occupied_cells(self):
        grid = np.full((5, 5), EMPTY, dtype=np.int8)
        grid[2, 2] = Gem.WHITE   # center already occupied
        b = Board(grid)
        bag = {'red': 1, 'white': 0, 'black': 0,
               'blue': 0, 'green': 0, 'pearl': 0, 'gold': 0}
        new_b, _, _ = b.refill(bag)
        # Center must still be WHITE, not overwritten
        assert new_b.token_at(2, 2) == Gem.WHITE

    def test_has_gold(self):
        grid = np.full((5, 5), EMPTY, dtype=np.int8)
        grid[0, 0] = Gem.GOLD
        b = Board(grid)
        assert b.has_gold() is True
        assert Board.empty().has_gold() is False