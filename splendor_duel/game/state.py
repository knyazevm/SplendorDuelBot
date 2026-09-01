"""
state.py — Complete game state.

Ties together Board, PlayerState, card pyramid, and phase tracking.
Immutable-style: engine.apply_action returns a new GameState.
"""
from __future__ import annotations

import random
from typing import Optional

from .actions import Phase
from .board import Board
from .card import Card, RoyalCard, load_cards
from .constants import (
    GEM_NAMES,
    MAX_SCROLLS,
    N_ROYAL_CARDS,
    PYRAMID_OPEN,
    TOKEN_COUNTS,
)
from .player import PlayerState


class GameState:
    __slots__ = (
        'board', 'bag', 'pyramid', 'decks', 'royal_cards',
        'scrolls_center', 'players', 'current_player', 'phase',
        'turn', 'pending_effect', 'pending_card', 'extra_turn_flag',
        'winner',
    )

    def __init__(
            self,
            board: Board,
            bag: dict[str, int],
            pyramid: dict[int, list[Card]],
            decks: dict[int, list[Card]],
            royal_cards: list[RoyalCard],
            scrolls_center: int,
            players: tuple[PlayerState, PlayerState],
            current_player: int,
            phase: Phase,
            turn: int = 1,
            pending_effect: Optional[str] = None,
            pending_card: Optional[Card] = None,
            extra_turn_flag: bool = False,
            winner: Optional[int] = None,
    ) -> None:
        self.board = board
        self.bag = bag
        self.pyramid = pyramid
        self.decks = decks
        self.royal_cards = royal_cards
        self.scrolls_center = scrolls_center
        self.players = players
        self.current_player = current_player
        self.phase = phase
        self.turn = turn
        self.pending_effect = pending_effect
        self.pending_card = pending_card
        self.extra_turn_flag = extra_turn_flag
        # Recorded by the engine when it sets GAME_OVER, never recomputed.
        self.winner = winner

    # ── Setup ─────────────────────────────────────────────────────────────────

    @classmethod
    def new_game(cls, cards_path: str) -> GameState:
        """
        Create a fresh game state following the official setup rules.

        1. Shuffle 3 decks, deal pyramid (5 / 4 / 3 visible cards).
        2. Fill board with all 25 tokens from bag.
        3. Place 3 scrolls.  Lay out 4 royal cards.
        4. Pick first player randomly; second player gets 1 scroll.
        """
        all_cards, all_royals = load_cards(cards_path)

        # ── Separate cards by level, shuffle each deck ────────────────────────
        decks: dict[int, list[Card]] = {1: [], 2: [], 3: []}
        for card in all_cards:
            decks[card.level].append(card)
        for lvl in decks:
            random.shuffle(decks[lvl])

        # ── Deal visible pyramid ──────────────────────────────────────────────
        pyramid: dict[int, list[Card]] = {}
        for lvl, n_open in PYRAMID_OPEN.items():
            pyramid[lvl] = decks[lvl][:n_open]
            decks[lvl] = decks[lvl][n_open:]

        # ── Royal cards ───────────────────────────────────────────────────────
        assert len(all_royals) == N_ROYAL_CARDS
        royal_cards = list(all_royals)

        # ── Board ─────────────────────────────────────────────────────────────
        board, bag = Board.initial()
        # After initial setup, bag should be empty (25 tokens on 25 cells)

        # ── Players ──────────────────────────────────────────────────────────
        p0 = PlayerState()
        p1 = PlayerState()

        # Random first player; second player gets 1 scroll to compensate
        first = random.randint(0, 1)
        second = 1 - first
        [p0, p1][second].scrolls = 1

        scrolls_center = MAX_SCROLLS - 1  # 1 given to second player

        return cls(
            board=board,
            bag=bag,
            pyramid=pyramid,
            decks=decks,
            royal_cards=royal_cards,
            scrolls_center=scrolls_center,
            players=(p0, p1),
            current_player=first,
            phase=Phase.OPTIONAL,
            turn=1,
        )

    # ── Copy ──────────────────────────────────────────────────────────────────

    def copy(self) -> GameState:
        return GameState(
            board=self.board.copy(),
            bag=dict(self.bag),
            pyramid={lvl: list(cards) for lvl, cards in self.pyramid.items()},
            decks={lvl: list(cards) for lvl, cards in self.decks.items()},
            royal_cards=list(self.royal_cards),
            scrolls_center=self.scrolls_center,
            players=(self.players[0].copy(), self.players[1].copy()),
            current_player=self.current_player,
            phase=self.phase,
            turn=self.turn,
            pending_effect=self.pending_effect,
            pending_card=self.pending_card,
            extra_turn_flag=self.extra_turn_flag,
            winner=self.winner,
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def active(self) -> PlayerState:
        """Current player."""
        return self.players[self.current_player]

    @property
    def opponent(self) -> PlayerState:
        """Opponent of current player."""
        return self.players[1 - self.current_player]

    @property
    def is_game_over(self) -> bool:
        return self.phase == Phase.GAME_OVER

    # `winner` is a plain attribute, set once by the engine at GAME_OVER — not
    # a property that re-derives the answer on every read.  It used to scan the
    # players in index order and return the first one satisfying a victory
    # condition, so a tie resolved as "player 0 wins", and any terminal state
    # reached without a victory condition silently reported None.  Callers then
    # had to guess what None meant, and mostly guessed wrong: it read as a loss
    # for whoever asked, or as a win for player 0.
    #
    # None now means exactly one thing — the game ended with no winner — and it
    # can only be produced deliberately.

    # ── Pyramid helpers ───────────────────────────────────────────────────────

    def pyramid_card(self, level: int, index: int) -> Card:
        return self.pyramid[level][index]

    def refill_pyramid_slot(self, level: int, index: int) -> None:
        """
        Replace the taken pyramid card with the top of the matching deck.
        Mutates in-place (caller is expected to be working on a copy).
        """
        if self.decks[level]:
            self.pyramid[level][index] = self.decks[level].pop(0)
        else:
            # Deck exhausted — remove the empty slot
            self.pyramid[level].pop(index)

    # ── Scroll helpers ────────────────────────────────────────────────────────

    def give_scroll_to(self, player_idx: int) -> None:
        """
        Give a scroll to the specified player, following the rules:
        - Take from center if available.
        - If center is empty, take from opponent.
        - If all 3 scrolls belong to one player, nothing happens.
        """
        if self.scrolls_center > 0:
            self.scrolls_center -= 1
            self.players[player_idx].scrolls += 1
        else:
            other = 1 - player_idx
            if self.players[other].scrolls > 0:
                self.players[other].scrolls -= 1
                self.players[player_idx].scrolls += 1
            # else: all 3 scrolls already with this player — do nothing

    def return_scroll_to_center(self, player_idx: int) -> None:
        """Return 1 scroll from player to center (when using a scroll)."""
        assert self.players[player_idx].scrolls > 0
        self.players[player_idx].scrolls -= 1
        self.scrolls_center += 1

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        lines = [
            f"=== Turn {self.turn}  Player {self.current_player}  "
            f"Phase {self.phase.name} ===",
            f"Scrolls in center: {self.scrolls_center}",
            f"Bag: {sum(self.bag.values())} tokens remaining",
            f"Royal cards available: {len(self.royal_cards)}",
            "",
            "Board:",
            str(self.board),
            "",
            f"Pyramid L1 ({len(self.pyramid.get(1, []))} cards): "
            f"{[c.id for c in self.pyramid.get(1, [])]}",
            f"Pyramid L2 ({len(self.pyramid.get(2, []))} cards): "
            f"{[c.id for c in self.pyramid.get(2, [])]}",
            f"Pyramid L3 ({len(self.pyramid.get(3, []))} cards): "
            f"{[c.id for c in self.pyramid.get(3, [])]}",
            "",
            f"Player 0: {self.players[0]}",
            f"Player 1: {self.players[1]}",
        ]
        if self.pending_effect:
            lines.append(f"Pending effect: {self.pending_effect}")
        return '\n'.join(lines)
