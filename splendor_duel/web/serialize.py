"""
serialize.py — State and action serialization to JSON-safe dicts.

Shared between game server and replay export.
"""
from __future__ import annotations

from splendor_duel.game.actions import (
    Action, BuyCard, ChooseRoyal, DiscardToken,
    EffectChooseWildcard, EffectSkip, EffectTakeOpponentGem,
    EffectTakeSameGem, ProceedToMain, RefillBoard,
    ReserveCard, TakeTokens, UseScroll,
)
from splendor_duel.game.constants import GEM_NAMES, N_GEMS
from splendor_duel.game.state import GameState


def card_to_dict(card) -> dict:
    return {
        "id": card.id, "level": card.level, "points": card.points,
        "crowns": card.crowns, "ability": card.ability,
        "is_wildcard": card.is_wildcard,
        "cost": {GEM_NAMES[i]: int(card.cost[i]) for i in range(N_GEMS)},
        "gem_bonus": (
            None if card.gem_bonus is None
            else {GEM_NAMES[i]: int(card.gem_bonus[i])
                  for i in range(N_GEMS) if card.gem_bonus[i] > 0}
        ),
    }


def player_to_dict(p) -> dict:
    return {
        "tokens": {GEM_NAMES[i]: int(p.tokens[i]) for i in range(N_GEMS)},
        "bonuses": {GEM_NAMES[i]: int(p.bonuses[i]) for i in range(N_GEMS)},
        "points": p.points, "crowns": p.crowns, "scrolls": p.scrolls,
        "cards": [card_to_dict(c) for c in p.cards],
        "reserved": [card_to_dict(c) for c in p.reserved],
        "royals": [{"id": r.id, "points": r.points, "ability": r.ability}
                   for r in p.royals],
        "total_tokens": int(p.total_tokens),
        "wildcard_assignments": {
            cid: int(gem) for cid, gem in p.wildcard_assignments.items()
        },
    }


def state_to_dict(state: GameState) -> dict:
    board = state.board.grid
    return {
        "board": [[int(board[r, c]) for c in range(5)] for r in range(5)],
        "bag_total": sum(state.bag.values()),
        "pyramid": {
            str(lvl): [card_to_dict(c) for c in cards]
            for lvl, cards in state.pyramid.items()
        },
        "deck_sizes": {str(lvl): len(d) for lvl, d in state.decks.items()},
        "royal_cards": [{"id": r.id, "points": r.points, "ability": r.ability}
                        for r in state.royal_cards],
        "scrolls_center": state.scrolls_center,
        "players": [player_to_dict(p) for p in state.players],
        "current_player": state.current_player,
        "phase": state.phase.name,
        "turn": state.turn,
        "is_game_over": state.is_game_over,
        "winner": state.winner,
    }


def serialize_action(action: Action, state: GameState) -> dict:
    if isinstance(action, TakeTokens):
        gems = []
        for r, c in action.positions:
            gem = state.board.token_at(r, c)
            gems.append(GEM_NAMES[gem] if gem is not None else "?")
        return {"type": "TakeTokens",
                "positions": [[r, c] for r, c in action.positions],
                "gems": gems, "desc": f"Take {'+'.join(gems)}"}
    if isinstance(action, BuyCard):
        card = (state.pyramid[action.level][action.index]
                if action.source == "pyramid"
                else state.active.reserved[action.index])
        return {"type": "BuyCard", "source": action.source,
                "level": action.level, "index": action.index,
                "card_id": card.id, "desc": f"Buy {card.id} ({card.points}pts)"}
    if isinstance(action, ReserveCard):
        cid = (state.pyramid[action.level][action.index].id
               if action.source == "pyramid"
               else f"L{action.level} deck")
        return {"type": "ReserveCard", "source": action.source,
                "level": action.level, "index": action.index,
                "card_id": cid, "desc": f"Reserve {cid} +gold"}
    if isinstance(action, UseScroll):
        gem = state.board.token_at(*action.position)
        gn = GEM_NAMES[gem] if gem is not None else "?"
        return {"type": "UseScroll", "position": list(action.position),
                "gem": gn, "desc": f"Scroll → take {gn}"}
    if isinstance(action, RefillBoard):
        return {"type": "RefillBoard", "desc": "Refill board"}
    if isinstance(action, ProceedToMain):
        return {"type": "ProceedToMain", "desc": "Proceed →"}
    if isinstance(action, EffectTakeSameGem):
        gem = state.board.token_at(*action.position)
        gn = GEM_NAMES[gem] if gem is not None else "?"
        return {"type": "EffectTakeSameGem", "position": list(action.position),
                "gem": gn, "desc": f"Take matching {gn}"}
    if isinstance(action, EffectTakeOpponentGem):
        return {"type": "EffectTakeOpponentGem", "gem": GEM_NAMES[action.gem],
                "gem_index": action.gem, "desc": f"Steal {GEM_NAMES[action.gem]}"}
    if isinstance(action, EffectChooseWildcard):
        target = state.active.cards[action.target_card_index]
        return {"type": "EffectChooseWildcard",
                "target_card_index": action.target_card_index,
                "target_card_id": target.id, "desc": f"Wildcard → {target.id}"}
    if isinstance(action, EffectSkip):
        return {"type": "EffectSkip", "desc": "Skip (no targets)"}
    if isinstance(action, ChooseRoyal):
        royal = state.royal_cards[action.index]
        return {"type": "ChooseRoyal", "index": action.index,
                "royal_id": royal.id, "desc": f"Choose {royal.id} ({royal.points}pts)"}
    if isinstance(action, DiscardToken):
        return {"type": "DiscardToken", "gem": GEM_NAMES[action.gem],
                "gem_index": action.gem, "desc": f"Discard {GEM_NAMES[action.gem]}"}
    return {"type": "Unknown", "desc": str(action)}
