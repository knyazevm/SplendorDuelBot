"""
az_agent_v2.py — Play-time agent for networks trained by `scripts/train_az_v2.py`.

Three differences from `AZAgent`:

  * Loads through `load_az_network`, so a tanh-value checkpoint is rebuilt as
    an `AZNetwork`. Reading one with the plain loader silently drops the tanh
    and misreads every leaf evaluation.
  * Determinizes the hidden decks before searching, matching training and
    denying the search knowledge of the next refill.
  * Does not run a search it cannot use. `AZAgent` searched on every forced
    `ProceedToMain` and every DISCARD, then discovered the resulting move was
    not in the caller's `legal_actions` and fell back to `legal_actions[0]` —
    a full search thrown away, several times per turn.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from splendor_duel.agents.base_agent import BaseAgent
from splendor_duel.env.action_map import action_to_index, index_to_action
from splendor_duel.game.actions import Action, DiscardToken, Phase, ProceedToMain
from splendor_duel.game.constants import Gem, N_GEMS
from splendor_duel.game.state import GameState

from .mcts_az import NetworkEvaluator, run_mcts
from .network_az import load_az_network
from .self_play_v2 import adaptive_sims, determinize


class AZAgentV2(BaseAgent):

    def __init__(
        self,
        network,
        sims_per_action: int = 12,
        min_sims: int = 48,
        max_sims: int = 600,
        c_puct: float = 1.5,
        top_k: int = 0,
        device: str = "cpu",
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or f"AZv2({sims_per_action}/act)")
        self.network = network
        self.network.eval()
        self.sims_per_action = sims_per_action
        self.min_sims = min_sims
        self.max_sims = max_sims
        self.c_puct = c_puct
        self.top_k = top_k
        self._evaluator = NetworkEvaluator(network, device=device)
        self._rng = random.Random(seed)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "AZAgentV2":
        network, meta = load_az_network(path, device=device)
        it = meta.get("total_iterations", "?")
        kwargs.setdefault("name", f"AZv2(it{it})")
        return cls(network=network, device=device, **kwargs)

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Phases the search auto-resolves. Answer them the same way training
        # did, rather than searching a position whose answer gets discarded.
        if state.phase == Phase.OPTIONAL and all(
            isinstance(a, ProceedToMain) for a in legal_actions
        ):
            return legal_actions[0]
        if state.phase == Phase.DISCARD:
            return self._greedy_discard(state, legal_actions)

        n_legal = len(legal_actions)
        visits, _root, _resolved = run_mcts(
            determinize(state, self._rng),
            self._evaluator,
            n_simulations=adaptive_sims(
                n_legal, self.sims_per_action, self.min_sims, self.max_sims,
            ),
            c_puct=self.c_puct,
            dirichlet_eps=0.0,
            top_k=self.top_k,
        )
        if visits.sum() <= 0:
            return legal_actions[0]

        # Rank by visits and take the best action the caller actually offers.
        for idx in np.argsort(-visits):
            if visits[idx] <= 0:
                break
            for a in legal_actions:
                if action_to_index(a) == int(idx):
                    return a
        return legal_actions[0]

    @staticmethod
    def _greedy_discard(state: GameState, legal_actions: list[Action]) -> Action:
        """Discard the largest non-gold stack — identical to the rule
        `mcts_az._auto_resolve_trivial` applies inside the search."""
        p = state.active
        best_gem, best_count = int(Gem.GOLD), -1
        for g in range(N_GEMS):
            if g != Gem.GOLD and p.tokens[g] > best_count:
                best_count, best_gem = int(p.tokens[g]), g
        for a in legal_actions:
            if isinstance(a, DiscardToken) and a.gem == best_gem:
                return a
        return legal_actions[0]
