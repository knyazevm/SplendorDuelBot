"""
Tests for engine.py — legal actions + apply_action.
Run with:  pytest tests/test_engine.py -v
"""
import json
import numpy as np
import pytest

from splendor_duel.game.actions import (
    BuyCard, ChooseRoyal, DiscardToken, EffectChooseWildcard,
    EffectSkip, EffectTakeOpponentGem, EffectTakeSameGem,
    PassTurn, Phase, ProceedToMain, RefillBoard, ReserveCard,
    TakeTokens, UseScroll,
)
from splendor_duel.game.board import Board
from splendor_duel.game.card import Card, RoyalCard, load_cards
from splendor_duel.game.constants import (
    ABILITY_EXTRA_TURN, ABILITY_TAKE_OPPONENT_GEM,
    ABILITY_TAKE_SAME_GEM, ABILITY_TAKE_SCROLL,
    EMPTY, GEM_NAMES, Gem, MAX_RESERVED, MAX_TOKENS, N_GEMS,
)
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.player import PlayerState
from splendor_duel.game.state import GameState

CARDS_PATH = "data/cards.json"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_card(
        card_id="T_01", level=1, points=0, crowns=0, ability=None,
        cost_white=0, bonus_gem=Gem.WHITE, bonus_count=1, is_wildcard=False,
):
    cost = tuple(cost_white if i == Gem.WHITE else 0 for i in range(N_GEMS))
    if is_wildcard:
        bonus = None
    elif bonus_count == 0:
        bonus = None
    else:
        bonus = tuple(bonus_count if i == bonus_gem else 0 for i in range(N_GEMS))
    return Card(
        id=card_id, level=level, cost=cost,
        gem_bonus=bonus, is_wildcard=is_wildcard,
        points=points, crowns=crowns, ability=ability,
    )


def _minimal_state(
        phase=Phase.OPTIONAL,
        current_player=0,
        board_gems=None,
        bag=None,
        pyramid=None,
        decks=None,
        royal_cards=None,
        scrolls_center=2,
) -> GameState:
    """Create a minimal controllable game state for testing."""
    if board_gems is None:
        grid = np.full((5, 5), -1, dtype=np.int8)
        grid[0, 0] = Gem.RED
        grid[0, 1] = Gem.BLUE
        grid[0, 2] = Gem.GREEN
        grid[2, 2] = Gem.GOLD
    else:
        grid = np.full((5, 5), -1, dtype=np.int8)
        for (r, c), gem in board_gems.items():
            grid[r, c] = gem
    board = Board(grid)

    if bag is None:
        bag = {name: 0 for name in GEM_NAMES}

    if pyramid is None:
        pyramid = {
            1: [_make_card("P1_0", 1, cost_white=1, bonus_gem=Gem.RED)],
            2: [_make_card("P2_0", 2, cost_white=3, bonus_gem=Gem.BLUE, points=2)],
            3: [],
        }
    if decks is None:
        decks = {1: [], 2: [], 3: []}

    if royal_cards is None:
        royal_cards = [
            RoyalCard(id="R_01", points=2, ability=ABILITY_TAKE_SCROLL),
            RoyalCard(id="R_02", points=3, ability=None),
        ]

    p0 = PlayerState()
    p1 = PlayerState()
    p1.scrolls = 1

    return GameState(
        board=board, bag=bag, pyramid=pyramid, decks=decks,
        royal_cards=royal_cards, scrolls_center=scrolls_center,
        players=(p0, p1), current_player=current_player,
        phase=phase,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL PHASE
# ══════════════════════════════════════════════════════════════════════════════

class TestOptionalPhase:

    def test_proceed_to_main_always_available(self):
        s = _minimal_state()
        actions = GameEngine.get_legal_actions(s)
        assert any(isinstance(a, ProceedToMain) for a in actions)

    def test_use_scroll_when_player_has_scroll(self):
        s = _minimal_state()
        s.players[0].scrolls = 1
        actions = GameEngine.get_legal_actions(s)
        scroll_actions = [a for a in actions if isinstance(a, UseScroll)]
        assert len(scroll_actions) > 0

    def test_no_scroll_actions_without_scroll(self):
        s = _minimal_state()
        s.players[0].scrolls = 0
        actions = GameEngine.get_legal_actions(s)
        assert not any(isinstance(a, UseScroll) for a in actions)

    def test_refill_when_bag_has_tokens(self):
        s = _minimal_state()
        s.bag['red'] = 3
        actions = GameEngine.get_legal_actions(s)
        assert any(isinstance(a, RefillBoard) for a in actions)

    def test_no_refill_when_bag_empty(self):
        s = _minimal_state()
        actions = GameEngine.get_legal_actions(s)
        assert not any(isinstance(a, RefillBoard) for a in actions)

    def test_apply_use_scroll(self):
        s = _minimal_state()
        s.players[0].scrolls = 1
        s.scrolls_center = 1

        s2 = GameEngine.apply_action(s, UseScroll(position=(0, 0)))
        assert s2.players[0].scrolls == 0
        assert s2.scrolls_center == 2  # returned to center
        assert s2.players[0].tokens[Gem.RED] == 1
        assert s2.board.is_empty(0, 0)
        assert s2.phase == Phase.OPTIONAL  # can still do more optionals

    def test_apply_refill_gives_opponent_scroll(self):
        s = _minimal_state()
        s.bag['red'] = 5
        s.current_player = 0

        s2 = GameEngine.apply_action(s, RefillBoard())
        assert s2.players[1].scrolls > s.players[1].scrolls or \
               s2.scrolls_center < s.scrolls_center
        assert s2.phase == Phase.MAIN  # after refill → MAIN

    def test_proceed_to_main_changes_phase(self):
        s = _minimal_state()
        s2 = GameEngine.apply_action(s, ProceedToMain())
        assert s2.phase == Phase.MAIN


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PHASE — TAKE TOKENS
# ══════════════════════════════════════════════════════════════════════════════

class TestTakeTokens:

    def test_take_single_token(self):
        s = _minimal_state(phase=Phase.MAIN)
        s2 = GameEngine.apply_action(s, TakeTokens(positions=((0, 0),)))
        assert s2.active.tokens[Gem.RED] == 0  # player changed
        assert s2.players[0].tokens[Gem.RED] == 1

    def test_take_three_triggers_scroll(self):
        """3 tokens of same colour → opponent gets scroll."""
        board_gems = {(0, 0): Gem.RED, (0, 1): Gem.RED, (0, 2): Gem.RED}
        s = _minimal_state(phase=Phase.MAIN, board_gems=board_gems)
        s.players[1].scrolls = 0  # reset
        s.scrolls_center = 3

        s2 = GameEngine.apply_action(s, TakeTokens(
            positions=((0, 0), (0, 1), (0, 2))
        ))
        # Opponent (player 1) should get a scroll
        assert s2.players[1].scrolls == 1

    def test_take_two_pearls_triggers_scroll(self):
        board_gems = {(0, 0): Gem.PEARL, (0, 1): Gem.PEARL}
        s = _minimal_state(phase=Phase.MAIN, board_gems=board_gems)
        s.players[1].scrolls = 0
        s.scrolls_center = 3

        s2 = GameEngine.apply_action(s, TakeTokens(
            positions=((0, 0), (0, 1))
        ))
        assert s2.players[1].scrolls == 1

    def test_after_take_next_player_turn(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        s2 = GameEngine.apply_action(s, TakeTokens(positions=((0, 0),)))
        assert s2.current_player == 1
        assert s2.phase == Phase.OPTIONAL


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PHASE — RESERVE
# ══════════════════════════════════════════════════════════════════════════════

class TestReserveCard:

    def test_reserve_from_pyramid(self):
        s = _minimal_state(phase=Phase.MAIN)
        card_id = s.pyramid[1][0].id

        s2 = GameEngine.apply_action(s, ReserveCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.players[0].tokens[Gem.GOLD] == 1
        assert len(s2.players[0].reserved) == 1
        assert s2.players[0].reserved[0].id == card_id

    def test_reserve_from_deck(self):
        s = _minimal_state(phase=Phase.MAIN)
        deck_card = _make_card("D1_0", 1)
        s.decks[1] = [deck_card]

        s2 = GameEngine.apply_action(s, ReserveCard(
            source='deck', level=1, index=0
        ))
        assert len(s2.players[0].reserved) == 1
        assert s2.players[0].reserved[0].id == "D1_0"
        assert len(s2.decks[1]) == 0

    def test_no_reserve_without_gold(self):
        board_gems = {(0, 0): Gem.RED}  # no gold on board
        s = _minimal_state(phase=Phase.MAIN, board_gems=board_gems)
        actions = GameEngine.get_legal_actions(s)
        assert not any(isinstance(a, ReserveCard) for a in actions)

    def test_no_reserve_at_max_reserved(self):
        s = _minimal_state(phase=Phase.MAIN)
        for i in range(3):
            s.players[0].reserved.append(_make_card(f"R_{i}"))
        actions = GameEngine.get_legal_actions(s)
        assert not any(isinstance(a, ReserveCard) for a in actions)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PHASE — BUY
# ══════════════════════════════════════════════════════════════════════════════

class TestBuyCard:

    def test_buy_from_pyramid(self):
        s = _minimal_state(phase=Phase.MAIN)
        s.players[0].tokens[Gem.WHITE] = 3
        card = s.pyramid[2][0]  # costs 3 white

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=2, index=0
        ))
        assert card in s2.players[0].cards
        assert s2.players[0].tokens[Gem.WHITE] == 0
        assert s2.bag['white'] == 3  # paid tokens go to bag

    def test_buy_from_reserve(self):
        s = _minimal_state(phase=Phase.MAIN)
        card = _make_card("RES_0", 1, cost_white=0, points=1)
        s.players[0].reserved = [card]

        s2 = GameEngine.apply_action(s, BuyCard(
            source='reserve', level=1, index=0
        ))
        assert card in s2.players[0].cards
        assert len(s2.players[0].reserved) == 0

    def test_cannot_buy_unaffordable(self):
        s = _minimal_state(phase=Phase.MAIN)
        # Player has 0 tokens, pyramid card costs 1 white
        actions = GameEngine.get_legal_actions(s)
        buy_p1 = [a for a in actions
                  if isinstance(a, BuyCard) and a.source == 'pyramid' and a.level == 1]
        assert len(buy_p1) == 0

    def test_buy_with_gold_substitution(self):
        s = _minimal_state(phase=Phase.MAIN)
        s.players[0].tokens[Gem.GOLD] = 1
        # Card costs 1 white, player has 0 white but 1 gold

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.players[0].tokens[Gem.GOLD] == 0
        assert s2.bag['gold'] == 1

    def test_buy_refills_pyramid(self):
        s = _minimal_state(phase=Phase.MAIN)
        s.players[0].tokens[Gem.WHITE] = 1
        deck_card = _make_card("D1_NEW", 1)
        s.decks[1] = [deck_card]

        old_id = s.pyramid[1][0].id
        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        # Old card is now in player's hand
        assert any(c.id == old_id for c in s2.players[0].cards)
        # New card from deck is in pyramid
        assert any(c.id == "D1_NEW" for c in s2.pyramid[1])


# ══════════════════════════════════════════════════════════════════════════════
# EFFECTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEffects:

    def test_extra_turn(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        card = _make_card("ET_0", 1, cost_white=0, ability=ABILITY_EXTRA_TURN)
        s.pyramid[1] = [card]

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        # Should be same player's turn again
        assert s2.current_player == 0
        assert s2.phase == Phase.OPTIONAL

    def test_take_scroll(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        card = _make_card("TS_0", 1, cost_white=0, ability=ABILITY_TAKE_SCROLL)
        s.pyramid[1] = [card]
        s.scrolls_center = 2

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.players[0].scrolls == 1
        assert s2.scrolls_center == 1

    def test_take_same_gem_effect(self):
        board_gems = {
            (0, 0): Gem.RED, (1, 1): Gem.WHITE,
            (2, 2): Gem.GOLD, (3, 3): Gem.WHITE,
        }
        s = _minimal_state(
            phase=Phase.MAIN, board_gems=board_gems, current_player=0,
        )
        card = _make_card(
            "TSG_0", 1, cost_white=0,
            bonus_gem=Gem.WHITE, ability=ABILITY_TAKE_SAME_GEM,
        )
        s.pyramid[1] = [card]

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        # Should be in EFFECT phase
        assert s2.phase == Phase.EFFECT
        assert s2.pending_effect == ABILITY_TAKE_SAME_GEM

        # Legal actions: white tokens at (1,1) and (3,3)
        actions = GameEngine.get_legal_actions(s2)
        assert all(isinstance(a, EffectTakeSameGem) for a in actions)
        positions = {a.position for a in actions}
        assert (1, 1) in positions
        assert (3, 3) in positions
        assert (0, 0) not in positions  # red, not white

        # Apply effect
        s3 = GameEngine.apply_action(s2, EffectTakeSameGem(position=(1, 1)))
        assert s3.players[0].tokens[Gem.WHITE] == 1

    def test_take_same_gem_no_targets_auto_skip(self):
        board_gems = {(0, 0): Gem.RED, (2, 2): Gem.GOLD}
        s = _minimal_state(
            phase=Phase.MAIN, board_gems=board_gems, current_player=0,
        )
        card = _make_card(
            "TSG_1", 1, cost_white=0,
            bonus_gem=Gem.BLUE, ability=ABILITY_TAKE_SAME_GEM,
        )
        s.pyramid[1] = [card]

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        # No blue on board → effect auto-skipped → next player's turn
        assert s2.current_player == 1

    def test_take_opponent_gem_effect(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        card = _make_card(
            "TOG_0", 1, cost_white=0,
            ability=ABILITY_TAKE_OPPONENT_GEM,
        )
        s.pyramid[1] = [card]
        s.players[1].tokens[Gem.RED] = 3
        s.players[1].tokens[Gem.BLUE] = 1

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.phase == Phase.EFFECT
        actions = GameEngine.get_legal_actions(s2)
        gems = {a.gem for a in actions if isinstance(a, EffectTakeOpponentGem)}
        assert Gem.RED in gems
        assert Gem.BLUE in gems
        assert Gem.GOLD not in gems

        # Take a red
        s3 = GameEngine.apply_action(s2, EffectTakeOpponentGem(gem=Gem.RED))
        assert s3.players[0].tokens[Gem.RED] == 1
        assert s3.players[1].tokens[Gem.RED] == 2

    def test_take_opponent_gem_no_targets(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        card = _make_card("TOG_1", 1, cost_white=0, ability=ABILITY_TAKE_OPPONENT_GEM)
        s.pyramid[1] = [card]
        # Opponent has no tokens → auto-skip

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.current_player == 1


# ══════════════════════════════════════════════════════════════════════════════
# WILDCARD
# ══════════════════════════════════════════════════════════════════════════════

class TestWildcard:

    def test_cannot_buy_wildcard_without_bonuses(self):
        s = _minimal_state(phase=Phase.MAIN)
        wc = _make_card("WC_0", 1, cost_white=0, is_wildcard=True)
        s.pyramid[1] = [wc]
        actions = GameEngine.get_legal_actions(s)
        buy_actions = [a for a in actions if isinstance(a, BuyCard)]
        assert len(buy_actions) == 0

    def test_buy_wildcard_enters_effect_phase(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        wc = _make_card("WC_0", 1, cost_white=0, is_wildcard=True, points=2)
        s.pyramid[1] = [wc]
        # Give player a card with bonus to place wildcard on
        base_card = _make_card("BASE_0", 1, bonus_gem=Gem.GREEN, bonus_count=1)
        s.players[0].add_card(base_card)

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.phase == Phase.EFFECT
        assert s2.pending_effect == 'choose_wildcard'

    def test_wildcard_choose_target(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        wc = _make_card("WC_0", 1, cost_white=0, is_wildcard=True, points=2)
        s.pyramid[1] = [wc]
        base_card = _make_card("BASE_0", 1, bonus_gem=Gem.GREEN, bonus_count=1)
        s.players[0].add_card(base_card)

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        actions = GameEngine.get_legal_actions(s2)
        assert all(isinstance(a, EffectChooseWildcard) for a in actions)

        # Choose the base card (index 0)
        s3 = GameEngine.apply_action(
            s2, EffectChooseWildcard(colour=int(Gem.GREEN)))
        assert s3.players[0].bonuses[Gem.GREEN] == 2  # 1 from base + 1 from wildcard
        assert s3.players[0].points == 2
        assert s3.current_player == 1  # turn ended

    def test_wildcard_is_worth_one_regardless_of_target(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        wc = _make_card("WC_0", 1, cost_white=0, is_wildcard=True)
        s.pyramid[1] = [wc]
        base_card = _make_card("BASE_0", 1, bonus_gem=Gem.BLACK, bonus_count=2)
        s.players[0].add_card(base_card)

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        s3 = GameEngine.apply_action(
            s2, EffectChooseWildcard(colour=int(Gem.BLACK)))
        # The wildcard grants 1 bonus, not a copy of the target's count.
        assert s3.players[0].bonuses[Gem.BLACK] == 3  # 2 from base + 1 from wildcard


# ══════════════════════════════════════════════════════════════════════════════
# ROYAL CARDS
# ══════════════════════════════════════════════════════════════════════════════

class TestPassTurn:
    """
    PassTurn exists only because the engine could otherwise return zero legal
    actions.  It must never be offered alongside a real move.
    """

    def _stuck_state(self):
        """
        Board holds only gold, bag empty, reserve full, nothing affordable.
        Reached in real play when privileges strip the last non-gold tokens.
        """
        s = GameState.new_game(CARDS_PATH)
        grid = np.full((5, 5), EMPTY, dtype=np.int8)
        grid[1, 3] = grid[2, 2] = grid[3, 2] = int(Gem.GOLD)
        s.board = Board(grid)
        s.bag = {name: 0 for name in GEM_NAMES}
        p = s.active
        p.scrolls = 0
        p.tokens = np.array([3, 0, 3, 3, 3, 0, 0], dtype=np.int8)  # 12: over the limit
        # Reserve full (so can_reserve is False) with cards it cannot afford:
        # 8 white against 3 white and no gold.
        p.reserved = [_make_card(f"EXP_{i}", 3, cost_white=8)
                      for i in range(MAX_RESERVED)]
        s.pyramid = {1: [], 2: [], 3: []}                          # nothing to buy
        s.phase = Phase.OPTIONAL
        return s

    def test_offered_when_nothing_else_is_possible(self):
        s = self._stuck_state()
        assert GameEngine.get_legal_actions(s) == [PassTurn()]

    def test_not_offered_when_any_other_action_exists(self):
        """Must move if you can — one takeable token is enough to remove it."""
        s = self._stuck_state()
        grid = np.array(s.board.grid, dtype=np.int8)
        grid[0, 0] = int(Gem.RED)
        s.board = Board(grid)
        actions = GameEngine.get_legal_actions(s)
        assert actions, "expected at least ProceedToMain"
        assert PassTurn() not in actions
        # ...and not in MAIN either, where a take is now available
        assert PassTurn() not in GameEngine.get_legal_actions(
            GameEngine.apply_action(s, ProceedToMain()))

    def test_never_offered_with_a_scroll_in_hand(self):
        s = self._stuck_state()
        s.active.scrolls = 1
        s.scrolls_center = 0
        grid = np.array(s.board.grid, dtype=np.int8)
        grid[4, 4] = int(Gem.BLUE)  # a scroll target
        s.board = Board(grid)
        assert PassTurn() not in GameEngine.get_legal_actions(s)

    def test_enforces_token_limit_and_ends_turn(self):
        s = self._stuck_state()
        me = s.current_player
        s2 = GameEngine.apply_action(s, PassTurn())
        # Over the limit, so the pass routes through DISCARD before ending.
        assert s2.phase == Phase.DISCARD and s2.current_player == me
        while s2.phase == Phase.DISCARD:
            s2 = GameEngine.apply_action(s2, GameEngine.get_legal_actions(s2)[0])
        assert s2.current_player == 1 - me, "turn should have passed"
        assert int(s2.players[me].tokens.sum()) == MAX_TOKENS

    def test_pass_breaks_the_deadlock(self):
        """
        The discard hands tokens back to the bag, so RefillBoard becomes legal
        again — two players cannot pass at each other forever.
        """
        s = self._stuck_state()
        assert sum(s.bag.values()) == 0
        s2 = GameEngine.apply_action(s, PassTurn())
        while s2.phase == Phase.DISCARD:
            s2 = GameEngine.apply_action(s2, GameEngine.get_legal_actions(s2)[0])
        assert sum(s2.bag.values()) > 0
        assert RefillBoard() in GameEngine.get_legal_actions(s2)


class TestRoyalCards:

    def test_crowns_trigger_royal_phase(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        card = _make_card("CR_0", 1, cost_white=0, crowns=3)
        s.pyramid[1] = [card]

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.phase == Phase.ROYAL
        assert s2.players[0].crowns == 3

    def test_choose_royal(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        card = _make_card("CR_0", 1, cost_white=0, crowns=3)
        s.pyramid[1] = [card]

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        actions = GameEngine.get_legal_actions(s2)
        assert all(isinstance(a, ChooseRoyal) for a in actions)
        assert len(actions) == 2  # two royal cards available

        s3 = GameEngine.apply_action(s2, ChooseRoyal(index=0))
        assert len(s3.players[0].royals) == 1
        assert s3.players[0].points == 2  # royal gives 2 pts

    def test_royal_with_take_scroll_ability(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        card = _make_card("CR_0", 1, cost_white=0, crowns=3)
        s.pyramid[1] = [card]
        s.scrolls_center = 2

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        # Royal R_01 has take_scroll ability
        s3 = GameEngine.apply_action(s2, ChooseRoyal(index=0))
        assert s3.players[0].scrolls == 1  # got scroll from royal


# ══════════════════════════════════════════════════════════════════════════════
# DISCARD
# ══════════════════════════════════════════════════════════════════════════════

class TestDiscard:

    def test_discard_phase_when_over_limit(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        s.players[0].tokens[Gem.RED] = 9
        # Take a token → now at 10, should be fine

        s2 = GameEngine.apply_action(s, TakeTokens(positions=((0, 0),)))
        # 9 red + 1 more = 10, exactly at limit
        assert s2.phase == Phase.OPTIONAL  # went to next player

    def test_discard_when_at_11(self):
        board_gems = {(0, 0): Gem.RED, (0, 1): Gem.BLUE}
        s = _minimal_state(phase=Phase.MAIN, board_gems=board_gems, current_player=0)
        s.players[0].tokens[Gem.RED] = 9

        s2 = GameEngine.apply_action(s, TakeTokens(positions=((0, 0), (0, 1))))
        # 9 + 2 = 11 → must discard
        assert s2.phase == Phase.DISCARD

        actions = GameEngine.get_legal_actions(s2)
        assert all(isinstance(a, DiscardToken) for a in actions)

        s3 = GameEngine.apply_action(s2, DiscardToken(gem=Gem.RED))
        assert s3.players[0].total_tokens == 10
        # After discarding → next player
        assert s3.current_player == 1


# ══════════════════════════════════════════════════════════════════════════════
# VICTORY
# ══════════════════════════════════════════════════════════════════════════════

class TestVictory:

    def test_victory_20_points(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        s.players[0].points = 19
        card = _make_card("WIN_0", 1, cost_white=0, points=1)
        s.pyramid[1] = [card]

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.phase == Phase.GAME_OVER
        assert s2.winner == 0

    def test_victory_10_crowns(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        s.players[0].crowns = 7  # needs royal which gives more?
        # Actually, let's just set high crowns and buy crown card
        s.players[0].crowns = 9
        s.players[0].royals = [
            RoyalCard("R_X", 2, None),
            RoyalCard("R_Y", 2, None),
        ]  # already has 2 royals so no royal phase
        card = _make_card("WIN_0", 1, cost_white=0, crowns=1)
        s.pyramid[1] = [card]

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.phase == Phase.GAME_OVER

    def test_no_premature_victory(self):
        s = _minimal_state(phase=Phase.MAIN, current_player=0)
        card = _make_card("NV_0", 1, cost_white=0, points=5)
        s.pyramid[1] = [card]

        s2 = GameEngine.apply_action(s, BuyCard(
            source='pyramid', level=1, index=0
        ))
        assert s2.phase != Phase.GAME_OVER


# ══════════════════════════════════════════════════════════════════════════════
# FULL GAME SIMULATION (random agents)
# ══════════════════════════════════════════════════════════════════════════════

FULL_CARDS_JSON = {
    "cards": [
        {
            "id": f"L{lvl}_{i:02d}", "level": lvl,
            "cost": {name: (1 if name == 'white' and lvl == 1
                            else 2 if name == 'white' and lvl == 2
            else 4 if name == 'white' and lvl == 3
            else 0)
                     for name in GEM_NAMES},
            "gem_bonus": {"white": 1} if i % 3 != 0 else {"red": 1},
            "points": lvl,
            "crowns": 1 if i % 4 == 0 else 0,
            "ability": None,
        }
        for lvl in (1, 2, 3)
        for i in range(10 if lvl == 1 else 8 if lvl == 2 else 6)
    ],
    "royal_cards": [
        {"id": "R_01", "points": 2, "ability": "take_scroll"},
        {"id": "R_02", "points": 2, "ability": "extra_turn"},
        {"id": "R_03", "points": 2, "ability": None},
        {"id": "R_04", "points": 3, "ability": None},
    ],
}


class TestFullGame:

    @pytest.fixture
    def cards_path(self, tmp_path):
        path = tmp_path / "cards.json"
        path.write_text(json.dumps(FULL_CARDS_JSON))
        return str(path)

    def test_random_game_terminates(self, cards_path):
        """Play a full game with random moves — must terminate."""
        import random
        random.seed(42)

        state = GameState.new_game(cards_path)
        max_steps = 2000
        steps = 0

        while not state.is_game_over and steps < max_steps:
            actions = GameEngine.get_legal_actions(state)
            assert len(actions) > 0, (
                f"No legal actions at turn {state.turn}, "
                f"phase {state.phase.name}, player {state.current_player}"
            )
            action = random.choice(actions)
            state = GameEngine.apply_action(state, action)
            steps += 1

        # Game should have ended
        assert state.is_game_over, (
            f"Game did not terminate after {max_steps} steps "
            f"(turn={state.turn}, phase={state.phase.name})"
        )
        assert state.winner in (0, 1)

    def test_multiple_random_games(self, cards_path):
        """Play 10 random games to check for crashes."""
        import random

        for seed in range(10):
            random.seed(seed)
            state = GameState.new_game(cards_path)
            steps = 0

            while not state.is_game_over and steps < 3000:
                actions = GameEngine.get_legal_actions(state)
                assert len(actions) > 0
                action = random.choice(actions)
                state = GameEngine.apply_action(state, action)
                steps += 1
