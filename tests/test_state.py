"""
Tests for actions.py, player.py, state.py.
Run with:  pytest tests/test_state.py -v
"""
import json
import numpy as np
import pytest

from splendor_duel.game.actions import (
    BuyCard, DiscardToken, Phase, ProceedToMain,
    TakeTokens, UseScroll,
)
from splendor_duel.game.card import Card, RoyalCard
from splendor_duel.game.constants import (
    CROWNS_ROYAL_1, CROWNS_ROYAL_2, CROWNS_WIN,
    GEM_NAMES, Gem, MAX_RESERVED, MAX_TOKENS,
    MONO_VP_WIN, N_GEMS, VP_WIN,
)
from splendor_duel.game.player import PlayerState
from splendor_duel.game.state import GameState

# ── Fixtures ──────────────────────────────────────────────────────────────────

CARDS_JSON = {
    "cards": [
        {
            "id": f"L{lvl}_{i:02d}", "level": lvl,
            "cost": {"white": lvl, "black": 0, "red": 0, "blue": 0,
                     "green": 0, "pearl": 0, "gold": 0},
            "gem_bonus": {"white": 1} if i % 2 == 0 else {"red": 1},
            "points": lvl, "crowns": 1 if i == 0 else 0, "ability": None,
        }
        for lvl in (1, 2, 3) for i in range(6 if lvl == 1 else 4 if lvl == 2 else 3)
    ],
    "royal_cards": [
        {"id": "R_01", "points": 2, "ability": "take_scroll"},
        {"id": "R_02", "points": 2, "ability": "extra_turn"},
        {"id": "R_03", "points": 2, "ability": None},
        {"id": "R_04", "points": 3, "ability": None},
    ],
}


@pytest.fixture
def cards_path(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(CARDS_JSON))
    return str(path)


def _make_card(
        card_id: str = "T_01",
        level: int = 1,
        cost_white: int = 0,
        bonus_gem: int = Gem.WHITE,
        bonus_count: int = 1,
        is_wildcard: bool = False,
        points: int = 0,
        crowns: int = 0,
        ability: str | None = None,
) -> Card:
    cost = tuple(cost_white if i == Gem.WHITE else 0 for i in range(N_GEMS))
    if is_wildcard:
        bonus = None
    else:
        bonus = tuple(bonus_count if i == bonus_gem else 0 for i in range(N_GEMS))
    return Card(
        id=card_id, level=level, cost=cost,
        gem_bonus=bonus, is_wildcard=is_wildcard,
        points=points, crowns=crowns, ability=ability,
    )


# ══════════════════════════════════════════════════════════════════════════════
# actions.py
# ══════════════════════════════════════════════════════════════════════════════

class TestActions:

    def test_actions_are_hashable(self):
        a1 = TakeTokens(positions=((0, 0), (0, 1)))
        a2 = TakeTokens(positions=((0, 0), (0, 1)))
        assert a1 == a2
        assert hash(a1) == hash(a2)
        assert len({a1, a2}) == 1

    def test_phase_ordering(self):
        assert Phase.OPTIONAL < Phase.MAIN < Phase.EFFECT < Phase.GAME_OVER


# ══════════════════════════════════════════════════════════════════════════════
# player.py
# ══════════════════════════════════════════════════════════════════════════════

class TestPlayerTokens:

    def test_initial_tokens_zero(self):
        p = PlayerState()
        assert p.total_tokens == 0

    def test_add_tokens(self):
        p = PlayerState()
        v = np.zeros(N_GEMS, dtype=np.int8)
        v[Gem.RED] = 3
        p.add_tokens(v)
        assert p.tokens[Gem.RED] == 3
        assert p.total_tokens == 3

    def test_remove_tokens(self):
        p = PlayerState()
        p.tokens[Gem.BLUE] = 5
        v = np.zeros(N_GEMS, dtype=np.int8)
        v[Gem.BLUE] = 2
        p.remove_tokens(v)
        assert p.tokens[Gem.BLUE] == 3

    def test_tokens_over_limit(self):
        p = PlayerState()
        p.tokens[Gem.RED] = 7
        p.tokens[Gem.BLUE] = 5
        assert p.tokens_over_limit == 2


class TestPlayerAffordability:

    def test_can_afford_free_card(self):
        p = PlayerState()
        card = _make_card(cost_white=0)
        assert p.can_afford(card)

    def test_cannot_afford_no_tokens(self):
        p = PlayerState()
        card = _make_card(cost_white=3)
        assert not p.can_afford(card)

    def test_afford_with_tokens(self):
        p = PlayerState()
        p.tokens[Gem.WHITE] = 3
        card = _make_card(cost_white=3)
        assert p.can_afford(card)

    def test_afford_with_bonuses(self):
        p = PlayerState()
        p.bonuses[Gem.WHITE] = 2
        p.tokens[Gem.WHITE] = 1
        card = _make_card(cost_white=3)
        assert p.can_afford(card)

    def test_afford_with_gold_substitution(self):
        p = PlayerState()
        p.tokens[Gem.WHITE] = 1
        p.tokens[Gem.GOLD] = 2
        card = _make_card(cost_white=3)
        assert p.can_afford(card)

    def test_not_enough_gold(self):
        p = PlayerState()
        p.tokens[Gem.GOLD] = 1
        card = _make_card(cost_white=3)
        assert not p.can_afford(card)

    def test_payment_vector_correct(self):
        p = PlayerState()
        p.tokens[Gem.WHITE] = 2
        p.tokens[Gem.GOLD] = 1
        p.bonuses[Gem.WHITE] = 1
        card = _make_card(cost_white=4)
        # need = max(4-1, 0) = 3 white;  pay 2 white + 1 gold
        payment = p.compute_payment(card)
        assert payment is not None
        assert payment[Gem.WHITE] == 2
        assert payment[Gem.GOLD] == 1


class TestPlayerCards:

    def test_add_card_updates_bonuses(self):
        p = PlayerState()
        card = _make_card(bonus_gem=Gem.RED, bonus_count=1)
        p.add_card(card)
        assert p.bonuses[Gem.RED] == 1
        assert len(p.cards) == 1

    def test_add_double_bonus_card(self):
        p = PlayerState()
        card = _make_card(bonus_gem=Gem.BLACK, bonus_count=2)
        p.add_card(card)
        assert p.bonuses[Gem.BLACK] == 2

    def test_add_wildcard_card(self):
        p = PlayerState()
        # Need existing bonus for wildcard
        p.bonuses[Gem.GREEN] = 1
        card = _make_card(is_wildcard=True, points=3)
        p.add_card(card, wildcard_color=Gem.GREEN)
        assert p.bonuses[Gem.GREEN] == 2  # +1 from wildcard
        assert card.id in p.wildcard_assignments

    def test_reserve_limit(self):
        p = PlayerState()
        for i in range(MAX_RESERVED):
            p.reserved.append(_make_card(card_id=f"R_{i}"))
        assert not p.can_reserve


class TestPlayerVictory:

    def test_no_victory_initially(self):
        p = PlayerState()
        assert p.check_victory() is None

    def test_victory_prestige_20(self):
        p = PlayerState()
        p.points = VP_WIN
        assert p.check_victory() == 'prestige_20'

    def test_victory_crowns_10(self):
        p = PlayerState()
        p.crowns = CROWNS_WIN
        assert p.check_victory() == 'crowns_10'

    def test_victory_mono_colour(self):
        p = PlayerState()
        for i in range(5):
            card = _make_card(
                card_id=f"M_{i}",
                bonus_gem=Gem.RED,
                points=2,
            )
            p.add_card(card)
        # 5 cards × 2 pts = 10 → mono red victory
        assert p.check_victory() == 'mono_red'

    def test_no_mono_victory_with_mixed_colours(self):
        p = PlayerState()
        for i, gem in enumerate([Gem.RED, Gem.RED, Gem.BLUE, Gem.BLUE]):
            p.add_card(_make_card(card_id=f"X_{i}", bonus_gem=gem, points=3))
        # red=6, blue=6 → neither reaches 10
        assert p.check_victory() is None

    def test_mono_victory_with_wildcard(self):
        p = PlayerState()
        p.bonuses[Gem.BLUE] = 1
        # Add 3 blue cards (3 pts each = 9)
        for i in range(3):
            p.add_card(_make_card(
                card_id=f"B_{i}", bonus_gem=Gem.BLUE, points=3,
            ))
        # Add 1 wildcard assigned to blue (2 pts → total 11)
        wc = _make_card(card_id="WC_1", is_wildcard=True, points=2)
        p.add_card(wc, wildcard_color=Gem.BLUE)
        assert p.check_victory() == 'mono_blue'

    def test_needs_royal_at_3_crowns(self):
        p = PlayerState()
        p.crowns = 3
        assert p.needs_royal()

    def test_needs_royal_at_6_crowns(self):
        p = PlayerState()
        p.crowns = 6
        p.royals.append(RoyalCard(id="R_01", points=2, ability=None))
        assert p.needs_royal()

    def test_no_royal_needed_with_2_royals(self):
        p = PlayerState()
        p.crowns = 8
        p.royals.append(RoyalCard(id="R_01", points=2, ability=None))
        p.royals.append(RoyalCard(id="R_02", points=2, ability=None))
        assert not p.needs_royal()


class TestPlayerCopy:

    def test_copy_is_independent(self):
        p = PlayerState()
        p.tokens[Gem.RED] = 5
        p.scrolls = 2
        p.cards.append(_make_card())

        p2 = p.copy()
        p2.tokens[Gem.RED] = 0
        p2.scrolls = 0
        p2.cards.pop()

        assert p.tokens[Gem.RED] == 5
        assert p.scrolls == 2
        assert len(p.cards) == 1


# ══════════════════════════════════════════════════════════════════════════════
# state.py
# ══════════════════════════════════════════════════════════════════════════════

class TestGameStateSetup:

    def test_new_game_creates_valid_state(self, cards_path):
        gs = GameState.new_game(cards_path)
        assert gs.phase == Phase.OPTIONAL
        assert gs.board.n_empty() == 0
        assert len(gs.royal_cards) == 4
        assert gs.current_player in (0, 1)
        assert gs.turn == 1

    def test_second_player_has_scroll(self, cards_path):
        gs = GameState.new_game(cards_path)
        total_scrolls = (
                gs.players[0].scrolls
                + gs.players[1].scrolls
                + gs.scrolls_center
        )
        assert total_scrolls == 3
        # Exactly one player has 1 scroll
        scroll_counts = [gs.players[0].scrolls, gs.players[1].scrolls]
        assert sorted(scroll_counts) == [0, 1]
        # The scroll must belong to the player who does NOT go first
        first = gs.current_player
        second = 1 - first
        assert gs.players[first].scrolls == 0, "First player should not have scroll"
        assert gs.players[second].scrolls == 1, "Second player should have scroll"

    def test_pyramid_sizes(self, cards_path):
        gs = GameState.new_game(cards_path)
        assert len(gs.pyramid[1]) == 5
        assert len(gs.pyramid[2]) == 4
        assert len(gs.pyramid[3]) == 3


class TestGameStateCopy:

    def test_copy_is_independent(self, cards_path):
        gs = GameState.new_game(cards_path)
        gs2 = gs.copy()

        gs2.players[0].tokens[Gem.RED] = 99
        gs2.scrolls_center = 0
        gs2.phase = Phase.GAME_OVER

        assert gs.players[0].tokens[Gem.RED] != 99
        assert gs.scrolls_center != 0
        assert gs.phase != Phase.GAME_OVER


class TestGameStateScrolls:

    def test_give_scroll_from_center(self, cards_path):
        gs = GameState.new_game(cards_path)
        gs = gs.copy()
        gs.scrolls_center = 2
        gs.players[0].scrolls = 0
        gs.give_scroll_to(0)
        assert gs.players[0].scrolls == 1
        assert gs.scrolls_center == 1

    def test_give_scroll_from_opponent(self, cards_path):
        gs = GameState.new_game(cards_path)
        gs = gs.copy()
        gs.scrolls_center = 0
        gs.players[0].scrolls = 0
        gs.players[1].scrolls = 2
        gs.give_scroll_to(0)
        assert gs.players[0].scrolls == 1
        assert gs.players[1].scrolls == 1

    def test_give_scroll_no_scrolls_anywhere(self, cards_path):
        gs = GameState.new_game(cards_path)
        gs = gs.copy()
        gs.scrolls_center = 0
        gs.players[0].scrolls = 3
        gs.players[1].scrolls = 0
        gs.give_scroll_to(0)
        # Already has all 3, nothing changes
        assert gs.players[0].scrolls == 3

    def test_return_scroll_to_center(self, cards_path):
        gs = GameState.new_game(cards_path)
        gs = gs.copy()
        gs.scrolls_center = 1
        gs.players[0].scrolls = 2
        gs.return_scroll_to_center(0)
        assert gs.players[0].scrolls == 1
        assert gs.scrolls_center == 2


class TestGameStateDisplay:

    def test_repr_does_not_crash(self, cards_path):
        gs = GameState.new_game(cards_path)
        s = repr(gs)
        assert 'Turn 1' in s
        assert 'Player 0' in s
