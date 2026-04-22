"""
mcts_agent.py — Pure Monte Carlo Tree Search (no neural network).

Optimised for Splendor Duel's high branching factor (~100 actions):
- Short rollouts (default 40 steps) + heuristic eval
- Progressive widening to avoid expanding all branches
- UCB1 with perspective-aware backpropagation

Usage:
    agent = MCTSAgent(iterations=200)
    action = agent.choose_action(state, legal_actions)
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
)
from splendor_duel.game.constants import Gem, N_GEMS, MONO_VP_WIN, VP_WIN, CROWNS_WIN
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState
from .base_agent import BaseAgent


# ── Heuristic evaluation ─────────────────────────────────────────────────────

def _evaluate(state: GameState, perspective: int) -> float:
    """
    Evaluate a non-terminal state as a win probability in [0, 1].

    Features:
    - Points (normalised toward 20)
    - Crowns (normalised toward 10)
    - Mono-colour progress (normalised toward 10)
    - Bonus count (economy strength)
    - Number of cards (engine efficiency)
    """
    me = state.players[perspective]
    opp = state.players[1 - perspective]

    def _score(p):
        s = 0.0
        # Points progress (max 20 needed)
        s += min(p.points / VP_WIN, 1.0) * 10.0
        # Crown progress (max 10)
        s += min(p.crowns / CROWNS_WIN, 1.0) * 3.0
        # Economy: bonuses reduce future costs
        s += min(int(p.bonuses.sum()) / 8.0, 1.0) * 2.0
        # Card count
        s += len(p.cards) * 0.3
        # Mono colour progress
        mono = p._mono_colour_points()
        if mono:
            best_mono = max(mono.values())
            s += min(best_mono / MONO_VP_WIN, 1.0) * 3.0
        # Royals
        s += len(p.royals) * 1.5
        return s

    my_score = _score(me)
    opp_score = _score(opp)

    # Sigmoid-style conversion to [0, 1]
    diff = my_score - opp_score
    return 1.0 / (1.0 + math.exp(-diff * 0.3))


# ── Tree node ─────────────────────────────────────────────────────────────────

class MCTSNode:
    """Single node in the MCTS tree."""
    __slots__ = (
        'state', 'parent', 'action', 'children',
        'untried_actions', 'visits', 'value_sum', 'player_who_acted',
    )

    def __init__(
        self,
        state: GameState,
        parent: Optional[MCTSNode] = None,
        action: Optional[Action] = None,
        player_who_acted: int = -1,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[MCTSNode] = []
        self.untried_actions: list[Action] = []
        self.visits: int = 0
        self.value_sum: float = 0.0  # sum of values from perspective of player_who_acted
        self.player_who_acted = player_who_acted

    @property
    def is_terminal(self) -> bool:
        return self.state.is_game_over


# ── MCTS agent ────────────────────────────────────────────────────────────────

class MCTSAgent(BaseAgent):
    """
    Pure MCTS agent optimised for Splendor Duel.

    Parameters:
        iterations:    number of tree iterations per move
        exploration:   UCB1 exploration constant
        rollout_depth: max steps in random rollout (default 40)
        time_limit:    optional wall-clock limit in seconds
        max_children:  progressive widening cap (0 = unlimited)
    """

    def __init__(
        self,
        iterations: int = 200,
        exploration: float = 1.41,
        rollout_depth: int = 40,
        time_limit: float = 0.0,
        max_children: int = 25,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=f"MCTS({iterations})")
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.time_limit = time_limit
        self.max_children = max_children
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
        # Shuffle to break ties randomly in progressive widening
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

            value = self._rollout(node.state, node.player_who_acted)
            self._backpropagate(node, value)

        if not root.children:
            return self._rng.choice(legal_actions)

        # Pick most-visited child
        best = max(root.children, key=lambda c: c.visits)
        return best.action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk down tree via UCB1 until expandable or terminal node."""
        while not node.is_terminal:
            # If there are untried actions and we haven't hit the width cap
            if node.untried_actions and (
                self.max_children == 0
                or len(node.children) < self.max_children
            ):
                return node
            if not node.children:
                return node  # terminal-like (no actions)
            node = self._best_ucb_child(node)
        return node

    def _best_ucb_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with best UCB1, adjusted for perspective."""
        current_player = node.state.current_player
        log_parent = math.log(node.visits) if node.visits > 0 else 0

        best_child = None
        best_score = -1e9

        for child in node.children:
            if child.visits == 0:
                return child  # always try unvisited
            # Value from perspective of current decision-maker
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
        """Expand one untried action."""
        action = node.untried_actions.pop()
        new_state = GameEngine.apply_action(node.state, action)
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action,
            player_who_acted=node.state.current_player,
        )
        if not new_state.is_game_over:
            actions = GameEngine.get_legal_actions(new_state)
            self._rng.shuffle(actions)
            child.untried_actions = actions
        node.children.append(child)
        return child

    def _rollout(self, state: GameState, last_player: int) -> float:
        """
        Fast rollout: skip trivial phases, random for rest.

        Returns value in [0, 1] from perspective of last_player.
        """
        s = state
        depth = 0
        rng_choice = self._rng.choice

        while not s.is_game_over and depth < self.rollout_depth:
            phase = s.phase

            # Fast-path: auto-resolve trivial phases without get_legal_actions
            if phase == Phase.OPTIONAL:
                s = GameEngine.apply_action(s, ProceedToMain())
                depth += 1
                continue

            if phase == Phase.DISCARD:
                # Discard most abundant non-gold
                p = s.active
                best_gem = -1
                best_count = -1
                for g in range(N_GEMS):
                    if g == Gem.GOLD:
                        continue
                    if p.tokens[g] > best_count:
                        best_count = p.tokens[g]
                        best_gem = g
                if best_gem >= 0:
                    s = GameEngine.apply_action(s, DiscardToken(gem=best_gem))
                else:
                    s = GameEngine.apply_action(s, DiscardToken(gem=int(Gem.GOLD)))
                depth += 1
                continue

            # MAIN, EFFECT, ROYAL — need legal actions
            actions = GameEngine.get_legal_actions(s)
            if not actions:
                break
            s = GameEngine.apply_action(s, rng_choice(actions))
            depth += 1

        if s.is_game_over:
            winner = s.winner
            if winner is None:
                return 0.5
            return 1.0 if winner == last_player else 0.0

        perspective = last_player if last_player >= 0 else 0
        return _evaluate(s, perspective)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Walk up tree, updating visits and value sums.

        `value` is from the perspective of the player who acted at the
        leaf node (the node passed in). Each ancestor stores value_sum
        from its own player_who_acted perspective.
        """
        leaf_player = node.player_who_acted
        while node is not None:
            node.visits += 1
            if node.player_who_acted >= 0:
                if node.player_who_acted == leaf_player:
                    node.value_sum += value
                else:
                    node.value_sum += (1.0 - value)
            node = node.parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_trivial(state: GameState, actions: list[Action]) -> bool:
    """Phases where MCTS overhead isn't worth it."""
    if state.phase == Phase.DISCARD:
        return True
    if state.phase == Phase.OPTIONAL:
        non_proceed = [a for a in actions if not isinstance(a, ProceedToMain)]
        if not non_proceed:
            return True
    return False


def _quick_pick(state: GameState, actions: list[Action], rng) -> Action:
    """Fast heuristic pick for trivial decisions."""
    # ProceedToMain
    for a in actions:
        if isinstance(a, ProceedToMain):
            return a
    # Discard: drop least useful (non-gold, most abundant)
    if state.phase == Phase.DISCARD:
        player = state.active
        discard_actions = [a for a in actions if isinstance(a, DiscardToken)]
        if discard_actions:
            def priority(a):
                if a.gem == Gem.GOLD:
                    return -100  # keep gold
                return int(player.tokens[a.gem])
            return max(discard_actions, key=priority)
    return rng.choice(actions)