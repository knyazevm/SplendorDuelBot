"""
mcts_az.py — AlphaZero-style MCTS.

Key differences from pure MCTS (mcts_agent.py):
  - PUCT selection (uses prior from policy head)
  - No rollouts — value comes from value head forward pass
  - All children expanded at once with priors
  - Dirichlet noise on root for self-play exploration

Entry points:
  run_mcts(state, network, n_simulations) → visit_counts[N_ACTIONS]
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from splendor_duel.game.actions import Action, Phase, ProceedToMain, DiscardToken
from splendor_duel.game.constants import Gem, N_GEMS
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState
from splendor_duel.env import N_ACTIONS, encode_state, legal_mask


# ── Node ──────────────────────────────────────────────────────────────────────

class AZNode:
    """
    AZ-MCTS node. Stores children keyed by action index.

    For a node representing state S where player P acts:
      - priors[a]:    P(a|S) from policy net for each legal action
      - visit_count[a]: N(S, a)
      - value_sum[a]:   W(S, a) — sum of backprop'd values from P's perspective
      - children[a]:    AZNode after action a (created lazily)
    """
    __slots__ = (
        'state', 'player_to_act', 'is_terminal', 'terminal_value',
        'priors', 'visit_count', 'value_sum', 'children',
        'legal_mask_np', 'legal_idx', 'legal_actions', 'total_visits', 'expanded',
    )

    def __init__(self, state: GameState):
        self.state = state
        self.is_terminal = state.is_game_over
        self.terminal_value: float = 0.0
        self.player_to_act: int = state.current_player if not self.is_terminal else -1
        self.children: dict[int, AZNode] = {}
        self.priors: Optional[np.ndarray] = None  # shape [N_ACTIONS]
        self.visit_count: Optional[np.ndarray] = None
        self.value_sum: Optional[np.ndarray] = None
        self.legal_mask_np: Optional[np.ndarray] = None
        self.legal_idx: Optional[np.ndarray] = None  # indices where mask is True
        self.legal_actions: Optional[list[tuple[int, Action]]] = None
        self.total_visits: int = 0
        self.expanded: bool = False

        if self.is_terminal:
            # Terminal value from perspective of winner (or 0 if draw)
            winner = state.winner
            if winner is None:
                self.terminal_value = 0.0
            else:
                # Stored as "who won" — lookup by perspective when backprop
                self.terminal_value = float(winner)


# ── Helpers for trivial phases ────────────────────────────────────────────────

def _auto_resolve_trivial(state: GameState) -> GameState:
    """
    Skip past phases where the choice is forced (no real decision).

    Returns advanced state. If state is already decision-worthy, returns as-is.
    """
    s = state
    max_iter = 20  # safety
    for _ in range(max_iter):
        if s.is_game_over:
            return s

        # OPTIONAL phase with only ProceedToMain → skip
        if s.phase == Phase.OPTIONAL:
            actions = GameEngine.get_legal_actions(s)
            if all(isinstance(a, ProceedToMain) for a in actions):
                s = GameEngine.apply_action(s, ProceedToMain())
                continue

        # DISCARD phase: pick largest non-gold stack (greedy, deterministic)
        if s.phase == Phase.DISCARD:
            p = s.active
            best_gem, best_count = -1, -1
            for g in range(N_GEMS):
                if g == Gem.GOLD:
                    continue
                if p.tokens[g] > best_count:
                    best_count = int(p.tokens[g])
                    best_gem = g
            if best_gem < 0:
                best_gem = int(Gem.GOLD)
            s = GameEngine.apply_action(s, DiscardToken(gem=best_gem))
            continue

        break
    return s


# ── Network wrapper ───────────────────────────────────────────────────────────

class NetworkEvaluator:
    """Wraps SplendorNetwork to provide (policy, value) eval on GameState."""

    def __init__(self, network, device='cpu'):
        self.network = network
        self.device = torch.device(device)
        self.network.eval()

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """
        Returns:
            policy: [N_ACTIONS] normalised probabilities over LEGAL actions
                    (illegal actions get 0 probability)
            value:  scalar in [-1, 1] from perspective of current_player
        """
        policy, value, _mask = self.evaluate_with_mask(state)
        return policy, value

    def evaluate_with_mask(
            self, state: GameState,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        As evaluate(), but also returns the legal mask it had to build anyway.

        Callers that need the mask should use this — building it means
        generating every legal action, which is one of the most expensive
        operations in the search.
        """
        obs = encode_state(state)
        mask = legal_mask(state)

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value_t = self.network(obs_t)
            logits = logits.squeeze(0).cpu().numpy()
            value = float(value_t.item())

        # Mask + softmax
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        exp_logits[~mask] = 0.0
        total = exp_logits.sum()
        if total < 1e-12:
            # Fallback: uniform over legal
            policy = mask.astype(np.float32)
            policy = policy / max(policy.sum(), 1.0)
        else:
            policy = exp_logits / total

        return policy.astype(np.float32), value, mask


# ── PUCT MCTS ─────────────────────────────────────────────────────────────────

def _select_child_puct(
        node: AZNode, c_puct: float,
) -> int:
    """PUCT selection: argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(sum N) / (1 + N(s,a)) ]

    Scores only the legal actions rather than all N_ACTIONS and masking after,
    which avoids allocating several N_ACTIONS-wide temporaries per simulation.
    """
    idx = node.legal_idx
    if idx.size == 0:
        return 0

    sqrt_total = math.sqrt(max(node.total_visits, 1))

    # Q values (mean value). value_sum is 0 wherever visits is 0, so the
    # clamped divide already yields the FPU of 0 for unvisited actions.
    visits = node.visit_count[idx]
    q = node.value_sum[idx] / np.maximum(visits, 1.0)
    u = (c_puct * sqrt_total) * node.priors[idx] / (1.0 + visits)

    return int(idx[np.argmax(q + u)])


def _apply_top_k(node: AZNode, top_k: int) -> None:
    """Restrict this node's search to its `top_k` highest-prior actions.

    Wide positions starve the search: a MAIN node with ~48 legal actions and
    ~234 simulations gets ~5 visits per action, and two independent searches
    from the same position then disagree by 2.969 nats — far more than the
    0.97 by which the network misses the target. Capping the branching spends
    the same simulations on fewer actions instead of spreading them to noise.
    (`mcts_agent.py` already does this with a ~25-child cap; the AZ search
    never inherited it.)

    Priors are renormalised over the survivors so the PUCT exploration term
    keeps its intended scale. `legal_mask_np` is deliberately left intact —
    it is the training target's mask and must stay the true legal set.
    """
    idx = node.legal_idx
    if top_k <= 0 or idx.size <= top_k:
        return
    keep = idx[np.argpartition(-node.priors[idx], top_k - 1)[:top_k]]
    node.legal_idx = np.sort(keep)
    total = float(node.priors[node.legal_idx].sum())
    if total > 1e-12:
        pruned = np.zeros_like(node.priors)
        pruned[node.legal_idx] = node.priors[node.legal_idx] / total
        node.priors = pruned


def _expand_node(
        node: AZNode, evaluator: NetworkEvaluator, top_k: int = 0,
) -> float:
    """
    Expand a leaf node: query network, store priors + mask, return value.
    Value returned is from perspective of node.player_to_act.
    """
    if node.is_terminal:
        # Terminal: return value from perspective of whoever acts at PARENT
        # Actually we need value from parent's perspective — but backprop
        # handles perspective flipping. Here just return winner info encoded
        # as +1/-1 from perspective of player_to_act (who can't act since game over).
        # Use a sentinel: if is_terminal, backprop handles it separately.
        return 0.0  # placeholder; backprop uses node.terminal_value

    policy, value, mask = evaluator.evaluate_with_mask(node.state)

    node.priors = policy
    node.visit_count = np.zeros(N_ACTIONS, dtype=np.float32)
    node.value_sum = np.zeros(N_ACTIONS, dtype=np.float32)
    node.legal_mask_np = mask
    node.legal_idx = np.flatnonzero(mask)
    node.expanded = True
    _apply_top_k(node, top_k)
    return value


def _backprop(path: list[tuple[AZNode, int]], leaf_value: float, leaf_player: int):
    """
    Walk back up the path, updating visit counts and value sums.

    `path`: list of (parent_node, action_taken) tuples from root to leaf's parent
    `leaf_value`: value at leaf, from perspective of leaf_player
    `leaf_player`: who acts at leaf (or -1 if terminal; handle separately)
    """
    for parent, action in reversed(path):
        # Update edge stats
        parent.visit_count[action] += 1
        parent.total_visits += 1

        # Value from parent.player_to_act's perspective
        if parent.player_to_act == leaf_player:
            parent.value_sum[action] += leaf_value
        else:
            parent.value_sum[action] += -leaf_value  # flip sign for 2-player zero-sum


def run_mcts(
        root_state: GameState,
        evaluator: NetworkEvaluator,
        n_simulations: int,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.0,  # 0 during play, 0.25 during self-play
        top_k: int = 0,  # 0 = unrestricted (original behaviour)
) -> tuple[np.ndarray, AZNode, GameState]:
    """
    Run AZ-MCTS from root_state.

    Returns:
        visit_counts: [N_ACTIONS] final visit counts at root
        root:         AZNode (for reuse or inspection)
        resolved_state: the state at root AFTER trivial phase resolution
                        (this is the state to apply argmax(visits) to)
    """
    # Advance past trivial phases before starting MCTS
    root_state = _auto_resolve_trivial(root_state)

    root = AZNode(root_state)
    if root.is_terminal:
        return np.zeros(N_ACTIONS, dtype=np.float32), root, root_state

    # Root is expanded WITHOUT pruning so Dirichlet noise below is applied over
    # the full legal set and can promote an action into the surviving top_k.
    _expand_node(root, evaluator)

    # Mix in Dirichlet noise on root priors (self-play only)
    if dirichlet_eps > 0:
        legal_indices = np.where(root.legal_mask_np)[0]
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal_indices))
        for i, a in enumerate(legal_indices):
            root.priors[a] = (1 - dirichlet_eps) * root.priors[a] + dirichlet_eps * noise[i]

    _apply_top_k(root, top_k)

    for _ in range(n_simulations):
        node = root
        path: list[tuple[AZNode, int]] = []

        # Selection — walk down via PUCT
        while node.expanded and not node.is_terminal:
            action = _select_child_puct(node, c_puct)

            if action in node.children:
                next_node = node.children[action]
            else:
                # Create child
                try:
                    next_state = GameEngine.apply_action(node.state, _index_to_action(action))
                except Exception:
                    # Defensive: mark action as explored but give it bad value
                    node.visit_count[action] += 1
                    node.total_visits += 1
                    node.value_sum[action] -= 1.0
                    node = None
                    break
                next_state = _auto_resolve_trivial(next_state)
                next_node = AZNode(next_state)
                node.children[action] = next_node

            path.append((node, action))
            node = next_node

        if node is None:
            continue

        # Expansion + Evaluation
        if node.is_terminal:
            # Terminal value: +1 if winner is player who acted last, -1 if opponent, 0 draw
            winner = node.state.winner
            if winner is None:
                leaf_value = 0.0
                leaf_player = path[-1][0].player_to_act if path else 0
            else:
                # leaf_player: who "would act" at terminal — use last acting player
                leaf_player = path[-1][0].player_to_act if path else winner
                leaf_value = 1.0 if winner == leaf_player else -1.0
        else:
            leaf_value = _expand_node(node, evaluator)
            leaf_player = node.player_to_act

        # Backprop
        _backprop(path, leaf_value, leaf_player)

    return root.visit_count.copy(), root, root_state


# ── Action index helper ──────────────────────────────────────────────────────

_action_cache = {}


def _index_to_action(index: int) -> Action:
    """Cached action lookup."""
    if index not in _action_cache:
        from splendor_duel.env.action_map import index_to_action
        _action_cache[index] = index_to_action(index)
    return _action_cache[index]
