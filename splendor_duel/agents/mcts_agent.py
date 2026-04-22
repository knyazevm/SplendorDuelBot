"""
mcts_agent.py — Improved MCTS for Splendor Duel.

Three key improvements over naive MCTS:

1. Rollout policy: 'random' (baseline), 'greedy' (uses GreedyAgent logic),
   or 'none' (pure heuristic eval, no rollout — fastest).

2. Action prioritization: untried actions are sorted by type priority
   (Buy > Reserve > best TakeTokens) so the tree expands promising
   branches first.

3. Configurable via constructor for easy tournament comparison:

   MCTSAgent(iterations=200, rollout='none')     # fast heuristic
   MCTSAgent(iterations=200, rollout='greedy')    # strong but slower
   MCTSAgent(iterations=500, rollout='random')    # baseline

Run tournaments:
   python scripts/run_tournament.py --include-mcts --games 30
"""
from __future__ import annotations

import math
import random
import time
from typing import Optional

import numpy as np

from splendor_duel.game.actions import (
    Action, BuyCard, Phase, ProceedToMain, DiscardToken,
    ReserveCard, TakeTokens, ChooseRoyal,
    EffectTakeSameGem, EffectTakeOpponentGem, EffectChooseWildcard,
    EffectSkip, RefillBoard, UseScroll,
)
from splendor_duel.game.constants import (
    Gem, N_GEMS, MONO_VP_WIN, VP_WIN, CROWNS_WIN, GEM_NAMES,
)
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState
from .base_agent import BaseAgent


# ══════════════════════════════════════════════════════════════════════════════
# HEURISTIC EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(state: GameState, perspective: int) -> float:
    """
    Evaluate a non-terminal state as a win probability in [0, 1].

    Richer than v1: adds token economy, affordable-card proximity,
    and reserved card value.
    """
    me = state.players[perspective]
    opp = state.players[1 - perspective]

    def _score(p, is_active):
        s = 0.0
        # Direct progress toward victory conditions
        s += min(p.points / VP_WIN, 1.0) * 12.0
        s += min(p.crowns / CROWNS_WIN, 1.0) * 4.0

        # Mono-colour progress (3rd win condition)
        mono = p._mono_colour_points()
        if mono:
            best_mono = max(mono.values())
            s += min(best_mono / MONO_VP_WIN, 1.0) * 4.0

        # Economy: bonuses are permanent cost reduction
        total_bonuses = int(p.bonuses.sum())
        s += min(total_bonuses / 8.0, 1.0) * 3.0

        # Cards bought (proxy for tempo)
        s += len(p.cards) * 0.4

        # Royals
        s += len(p.royals) * 2.0

        # Token wealth (having tokens = closer to buying)
        s += min(int(p.tokens.sum()) / 10.0, 1.0) * 1.0

        # Gold tokens are especially valuable
        s += min(int(p.tokens[Gem.GOLD]) / 3.0, 1.0) * 1.0

        # Scrolls
        s += p.scrolls * 0.5

        # Reserved cards (future purchases locked in)
        s += len(p.reserved) * 0.3

        return s

    my_score = _score(me, state.current_player == perspective)
    opp_score = _score(opp, state.current_player != perspective)

    diff = my_score - opp_score
    return 1.0 / (1.0 + math.exp(-diff * 0.25))


# ══════════════════════════════════════════════════════════════════════════════
# ACTION PRIORITIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _action_priority(action: Action, state: GameState) -> float:
    """
    Higher = expand first in MCTS tree.

    Buy > Reserve (good cards) > Take (useful tokens) > other.
    """
    if isinstance(action, BuyCard):
        if action.source == 'pyramid':
            card = state.pyramid[action.level][action.index]
        else:
            card = state.active.reserved[action.index]
        # Higher points/crowns = higher priority
        return 100.0 + card.points * 5.0 + card.crowns * 3.0

    if isinstance(action, ReserveCard):
        if action.source == 'pyramid':
            card = state.pyramid[action.level][action.index]
            return 50.0 + card.points * 3.0 + card.level * 2.0
        return 40.0 + action.level * 2.0  # blind reserve

    if isinstance(action, TakeTokens):
        n = len(action.positions)
        # Prefer 3-token takes, then 2, then 1
        return 20.0 + n * 3.0

    if isinstance(action, ChooseRoyal):
        royal = state.royal_cards[action.index]
        return 90.0 + royal.points * 3.0

    if isinstance(action, EffectTakeOpponentGem):
        return 80.0 + state.opponent.tokens[action.gem]

    if isinstance(action, EffectTakeSameGem):
        return 75.0

    if isinstance(action, EffectChooseWildcard):
        return 85.0

    if isinstance(action, ProceedToMain):
        return 10.0

    if isinstance(action, UseScroll):
        return 30.0

    return 5.0  # RefillBoard, EffectSkip, DiscardToken, etc.


def _sort_actions_by_priority(actions: list[Action], state: GameState) -> list[Action]:
    """Sort actions so highest-priority are popped last (expanded first)."""
    # We pop from the end, so sort ascending
    return sorted(actions, key=lambda a: _action_priority(a, state))


# ══════════════════════════════════════════════════════════════════════════════
# GREEDY ROLLOUT POLICY
# ══════════════════════════════════════════════════════════════════════════════

def _greedy_pick(actions: list[Action], state: GameState) -> Action:
    """
    Fast greedy action selection for rollouts.

    Simplified version of GreedyAgent — no numpy, no card_efficiency,
    just priority-based selection for speed.
    """
    # Buy the highest-value card
    best_buy = None
    best_buy_val = -1
    for a in actions:
        if isinstance(a, BuyCard):
            if a.source == 'pyramid':
                card = state.pyramid[a.level][a.index]
            else:
                card = state.active.reserved[a.index]
            val = card.points * 4 + card.crowns * 2.5
            if val > best_buy_val:
                best_buy_val = val
                best_buy = a
    if best_buy is not None:
        return best_buy

    # Take longest token line
    best_take = None
    best_take_len = 0
    for a in actions:
        if isinstance(a, TakeTokens):
            if len(a.positions) > best_take_len:
                best_take_len = len(a.positions)
                best_take = a
    if best_take is not None:
        return best_take

    # Royal: highest points
    royals = [a for a in actions if isinstance(a, ChooseRoyal)]
    if royals:
        return max(royals, key=lambda a: state.royal_cards[a.index].points)

    # Effect: take whatever is available
    for a in actions:
        if isinstance(a, (EffectTakeSameGem, EffectTakeOpponentGem)):
            return a

    # Fallback
    return actions[0]


# ══════════════════════════════════════════════════════════════════════════════
# TREE NODE
# ══════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = (
        'state', 'parent', 'action', 'children',
        'untried_actions', 'visits', 'value_sum', 'player_who_acted',
    )

    def __init__(self, state, parent=None, action=None, player_who_acted=-1):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[MCTSNode] = []
        self.untried_actions: list[Action] = []
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.player_who_acted = player_who_acted

    @property
    def is_terminal(self):
        return self.state.is_game_over


# ══════════════════════════════════════════════════════════════════════════════
# MCTS AGENT
# ══════════════════════════════════════════════════════════════════════════════

class MCTSAgent(BaseAgent):
    """
    Improved MCTS agent.

    Parameters:
        iterations:    tree iterations per move
        exploration:   UCB1 constant (default 1.41)
        rollout:       'random', 'greedy', or 'none' (heuristic only)
        rollout_depth: max steps in rollout (ignored if rollout='none')
        max_children:  progressive widening cap per node
        time_limit:    wall-clock limit in seconds (0 = use iterations)
        prioritize:    sort untried actions by strategic priority
    """

    def __init__(
            self,
            iterations: int = 200,
            exploration: float = 1.41,
            rollout: str = 'none',
            rollout_depth: int = 30,
            max_children: int = 20,
            time_limit: float = 0.0,
            prioritize: bool = True,
            seed: int | None = None,
    ) -> None:
        tag = f"MCTS({iterations},{rollout})"
        super().__init__(name=tag)
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_mode = rollout
        self.rollout_depth = rollout_depth
        self.max_children = max_children
        self.time_limit = time_limit
        self.prioritize = prioritize
        self._rng = random.Random(seed)
        self._player_index = 0

    def notify_game_start(self, player_index: int) -> None:
        self._player_index = player_index

    def choose_action(
            self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Skip MCTS for trivial phases
        if _is_trivial(state, legal_actions):
            return _quick_pick(state, legal_actions, self._rng)

        root = MCTSNode(state=state)
        if self.prioritize:
            root.untried_actions = _sort_actions_by_priority(legal_actions, state)
        else:
            untried = list(legal_actions)
            self._rng.shuffle(untried)
            root.untried_actions = untried

        deadline = (time.time() + self.time_limit) if self.time_limit > 0 else None
        iters = 0

        while True:
            if deadline:
                if time.time() >= deadline:
                    break
            elif iters >= self.iterations:
                break
            iters += 1

            node = self._select(root)

            if not node.is_terminal:
                if node.untried_actions and (
                        self.max_children == 0
                        or len(node.children) < self.max_children
                ):
                    node = self._expand(node)

            value = self._evaluate_node(node)
            self._backpropagate(node, value)

        if not root.children:
            return self._rng.choice(legal_actions)

        best = max(root.children, key=lambda c: c.visits)
        return best.action

    # ── Tree phases ───────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal:
            if node.untried_actions and (
                    self.max_children == 0
                    or len(node.children) < self.max_children
            ):
                return node
            if not node.children:
                return node
            node = self._best_ucb_child(node)
        return node

    def _best_ucb_child(self, node: MCTSNode) -> MCTSNode:
        current_player = node.state.current_player
        log_parent = math.log(node.visits) if node.visits > 0 else 0

        best_child = None
        best_score = -1e9

        for child in node.children:
            if child.visits == 0:
                return child
            if child.player_who_acted == current_player:
                win_rate = child.value_sum / child.visits
            else:
                win_rate = 1.0 - child.value_sum / child.visits
            explore = self.exploration * math.sqrt(log_parent / child.visits)
            score = win_rate + explore
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> MCTSNode:
        action = node.untried_actions.pop()  # pop highest-priority (end of sorted list)
        new_state = GameEngine.apply_action(node.state, action)
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action,
            player_who_acted=node.state.current_player,
        )
        if not new_state.is_game_over:
            actions = GameEngine.get_legal_actions(new_state)
            if self.prioritize:
                child.untried_actions = _sort_actions_by_priority(actions, new_state)
            else:
                self._rng.shuffle(actions)
                child.untried_actions = actions
        node.children.append(child)
        return child

    def _evaluate_node(self, node: MCTSNode) -> float:
        """Evaluate a leaf node using configured rollout mode."""
        if node.is_terminal:
            winner = node.state.winner
            if winner is None:
                return 0.5
            return 1.0 if winner == node.player_who_acted else 0.0

        if self.rollout_mode == 'none':
            # Pure heuristic — no rollout
            perspective = node.player_who_acted if node.player_who_acted >= 0 else 0
            return evaluate(node.state, perspective)

        # Rollout (random or greedy)
        return self._rollout(node.state, node.player_who_acted)

    def _rollout(self, state: GameState, last_player: int) -> float:
        s = state
        depth = 0
        use_greedy = (self.rollout_mode == 'greedy')
        rng_choice = self._rng.choice

        while not s.is_game_over and depth < self.rollout_depth:
            phase = s.phase

            # Fast-path: auto-resolve trivial phases
            if phase == Phase.OPTIONAL:
                s = GameEngine.apply_action(s, ProceedToMain())
                depth += 1
                continue

            if phase == Phase.DISCARD:
                p = s.active
                best_gem, best_count = -1, -1
                for g in range(N_GEMS):
                    if g == Gem.GOLD:
                        continue
                    if p.tokens[g] > best_count:
                        best_count = int(p.tokens[g])
                        best_gem = g
                if best_gem >= 0:
                    s = GameEngine.apply_action(s, DiscardToken(gem=best_gem))
                else:
                    s = GameEngine.apply_action(s, DiscardToken(gem=int(Gem.GOLD)))
                depth += 1
                continue

            actions = GameEngine.get_legal_actions(s)
            if not actions:
                break

            if use_greedy:
                action = _greedy_pick(actions, s)
            else:
                action = rng_choice(actions)

            s = GameEngine.apply_action(s, action)
            depth += 1

        if s.is_game_over:
            winner = s.winner
            if winner is None:
                return 0.5
            return 1.0 if winner == last_player else 0.0

        perspective = last_player if last_player >= 0 else 0
        return evaluate(s, perspective)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        leaf_player = node.player_who_acted
        while node is not None:
            node.visits += 1
            if node.player_who_acted >= 0:
                if node.player_who_acted == leaf_player:
                    node.value_sum += value
                else:
                    node.value_sum += (1.0 - value)
            node = node.parent


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _is_trivial(state: GameState, actions: list[Action]) -> bool:
    if state.phase == Phase.DISCARD:
        return True
    if state.phase == Phase.OPTIONAL:
        return all(isinstance(a, ProceedToMain) for a in actions)
    if state.phase == Phase.EFFECT:
        return len(actions) == 1 and isinstance(actions[0], EffectSkip)
    return False


def _quick_pick(state: GameState, actions: list[Action], rng) -> Action:
    for a in actions:
        if isinstance(a, ProceedToMain):
            return a
    if state.phase == Phase.DISCARD:
        player = state.active
        discard_actions = [a for a in actions if isinstance(a, DiscardToken)]
        if discard_actions:
            return max(discard_actions, key=lambda a: -100 if a.gem == Gem.GOLD else int(player.tokens[a.gem]))
    return rng.choice(actions)
