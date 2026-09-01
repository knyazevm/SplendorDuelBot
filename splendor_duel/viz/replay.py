"""
replay.py — Game recording and action descriptions.

Records full game history for visualization and analysis.
Provides human-readable action descriptions and JSON serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from splendor_duel.game.actions import (
    Action, BuyCard, ChooseRoyal, DiscardToken,
    EffectChooseGold, EffectChooseWildcard, EffectSkip, EffectTakeOpponentGem, PassTurn,
    EffectTakeSameGem, Phase, ProceedToMain, RefillBoard,
    ReserveCard, TakeTokens, UseScroll,
)
from splendor_duel.game.card import Card, RoyalCard
from splendor_duel.game.constants import GEM_NAMES, N_GEMS, Gem
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.player import PlayerState
from splendor_duel.game.state import GameState


# ── GameLog ───────────────────────────────────────────────────────────────────

@dataclass
class GameLog:
    """
    Full record of a played game.

    states[0]  = initial state
    actions[i] transforms states[i] → states[i+1]
    len(actions) == len(states) - 1
    """
    states: list[GameState] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    @property
    def is_finished(self) -> bool:
        return self.states and self.states[-1].is_game_over

    @property
    def winner(self) -> Optional[int]:
        if self.is_finished:
            return self.states[-1].winner
        return None

    def record(self, state: GameState, action: Optional[Action] = None) -> None:
        """Record a state (and optionally the action that led to next state)."""
        if not self.states:
            self.states.append(state)
        if action is not None:
            self.actions.append(action)
            new_state = GameEngine.apply_action(state, action)
            self.states.append(new_state)

    def add_initial(self, state: GameState) -> None:
        self.states = [state]
        self.actions = []

    def add_step(self, action: Action, resulting_state: GameState) -> None:
        self.actions.append(action)
        self.states.append(resulting_state)

    @classmethod
    def play_random_game(cls, cards_path: str, seed: int = 42) -> GameLog:
        """Play a full random game and return the log."""
        import random
        random.seed(seed)

        state = GameState.new_game(cards_path)
        log = cls()
        log.add_initial(state)

        while not state.is_game_over:
            actions = GameEngine.get_legal_actions(state)
            action = random.choice(actions)
            state = GameEngine.apply_action(state, action)
            log.add_step(action, state)

        return log


# ── Action descriptions ───────────────────────────────────────────────────────

GEM_SYMBOLS = {
    Gem.WHITE: '⬜', Gem.BLACK: '⬛', Gem.RED: '🔴',
    Gem.BLUE: '🔵', Gem.GREEN: '🟢', Gem.PEARL: '🫧', Gem.GOLD: '🟡',
}

GEM_LABELS = {
    Gem.WHITE: 'white', Gem.BLACK: 'black', Gem.RED: 'red',
    Gem.BLUE: 'blue', Gem.GREEN: 'green', Gem.PEARL: 'pearl', Gem.GOLD: 'gold',
}


def describe_action(action: Action, state: Optional[GameState] = None) -> str:
    """Human-readable description of an action."""
    if isinstance(action, UseScroll):
        r, c = action.position
        gem_name = ''
        if state:
            gem = state.board.token_at(r, c)
            gem_name = f' ({GEM_LABELS.get(gem, "?")})' if gem is not None else ''
        return f"Used scroll → took token{gem_name} from ({r},{c})"

    if isinstance(action, RefillBoard):
        return "Refilled board from bag"

    if isinstance(action, ProceedToMain):
        return "→ Main phase"

    if isinstance(action, TakeTokens):
        pos_str = ', '.join(f'({r},{c})' for r, c in action.positions)
        if state:
            gems = []
            for r, c in action.positions:
                gem = state.board.token_at(r, c)
                if gem is not None:
                    gems.append(GEM_LABELS.get(gem, '?'))
            return f"Took {'+'.join(gems)} from [{pos_str}]"
        return f"Took tokens from [{pos_str}]"

    if isinstance(action, ReserveCard):
        if action.source == 'pyramid':
            card_id = ''
            if state:
                card = state.pyramid.get(action.level, [])
                if action.index < len(card):
                    card_id = f' {card[action.index].id}'
            return f"Reserved{card_id} (L{action.level} pyramid) + took gold"
        return f"Reserved from L{action.level} deck + took gold"

    if isinstance(action, BuyCard):
        if action.source == 'pyramid':
            card_id = ''
            if state:
                card = state.pyramid.get(action.level, [])
                if action.index < len(card):
                    card_id = f' {card[action.index].id}'
            return f"Bought{card_id} from L{action.level} pyramid"
        else:
            card_id = ''
            if state:
                res = state.active.reserved
                if action.index < len(res):
                    card_id = f' {res[action.index].id}'
            return f"Bought{card_id} from reserve"

    if isinstance(action, PassTurn):
        return "Passed (no legal move)"

    if isinstance(action, EffectChooseWildcard):
        colour = GEM_LABELS.get(action.colour, '?')
        return f"Wildcard → {colour} bonus"

    if isinstance(action, EffectChooseGold):
        r, c = action.position
        return f"Took gold from ({r},{c})"

    if isinstance(action, EffectTakeSameGem):
        r, c = action.position
        return f"Effect: took matching gem from ({r},{c})"

    if isinstance(action, EffectTakeOpponentGem):
        return f"Effect: took {GEM_LABELS.get(action.gem, '?')} from opponent"

    if isinstance(action, EffectSkip):
        return "Effect: skipped (no valid targets)"

    if isinstance(action, ChooseRoyal):
        if state:
            royal = state.royal_cards[action.index]
            return f"Chose royal {royal.id} ({royal.points} pts)"
        return f"Chose royal card #{action.index}"

    if isinstance(action, DiscardToken):
        return f"Discarded {GEM_LABELS.get(action.gem, '?')} token"

    return str(action)


# ── State serialization (for HTML renderer) ───────────────────────────────────

def _card_to_dict(card: Card) -> dict:
    return {
        'id': card.id, 'level': card.level, 'points': card.points,
        'crowns': card.crowns, 'ability': card.ability,
        'is_wildcard': card.is_wildcard,
        'cost': {GEM_NAMES[i]: int(card.cost[i]) for i in range(N_GEMS)},
        'gem_bonus': (
            None if card.gem_bonus is None
            else {GEM_NAMES[i]: int(card.gem_bonus[i])
                  for i in range(N_GEMS) if card.gem_bonus[i] > 0}
        ),
    }


def _royal_to_dict(r: RoyalCard) -> dict:
    return {'id': r.id, 'points': r.points, 'ability': r.ability}


def _player_to_dict(p: PlayerState) -> dict:
    return {
        'tokens': {GEM_NAMES[i]: int(p.tokens[i]) for i in range(N_GEMS)},
        'bonuses': {GEM_NAMES[i]: int(p.bonuses[i]) for i in range(N_GEMS)},
        'points': p.points,
        'crowns': p.crowns,
        'scrolls': p.scrolls,
        'cards': [_card_to_dict(c) for c in p.cards],
        'reserved': [_card_to_dict(c) for c in p.reserved],
        'royals': [_royal_to_dict(r) for r in p.royals],
        'total_tokens': int(p.total_tokens),
        'wildcard_assignments': {card_id: int(gem_idx) for card_id, gem_idx in p.wildcard_assignments.items()},
    }


def state_to_dict(state: GameState) -> dict:
    """Serialize a GameState to a plain dict (JSON-safe)."""
    board = state.board.grid
    board_list = [[int(board[r, c]) for c in range(5)] for r in range(5)]

    return {
        'board': board_list,
        'bag_total': sum(state.bag.values()),
        'pyramid': {
            str(lvl): [_card_to_dict(c) for c in cards]
            for lvl, cards in state.pyramid.items()
        },
        'deck_sizes': {str(lvl): len(d) for lvl, d in state.decks.items()},
        'royal_cards': [_royal_to_dict(r) for r in state.royal_cards],
        'scrolls_center': state.scrolls_center,
        'players': [_player_to_dict(p) for p in state.players],
        'current_player': state.current_player,
        'phase': state.phase.name,
        'turn': state.turn,
    }


def log_to_json_data(log: GameLog) -> list[dict]:
    """
    Convert a GameLog to a list of step dicts for the HTML renderer.

    Each step has: state (serialized), action_description, step_index.
    """
    steps = []

    for i, state in enumerate(log.states):
        step = {
            'index': i,
            'state': state_to_dict(state),
            'description': '',
        }
        if i > 0:
            action = log.actions[i - 1]
            prev_state = log.states[i - 1]
            step['description'] = describe_action(action, prev_state)
            step['player_acted'] = prev_state.current_player
        steps.append(step)

    return steps
