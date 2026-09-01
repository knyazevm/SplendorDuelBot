"""
self_play_v2.py — Self-play game generation with the fixes described in
`scripts/train_az_v2.py`. Drop-in replacement for `self_play.generate_batch`.

Differences from `self_play.py`:

  1. Adaptive simulation budget. A flat 50 sims was spent equally on a
     2-legal-action EFFECT choice and a 139-action MAIN choice. Measured on
     the existing replay buffer, 53% of positions have <=10 legal actions and
     22% have >=30, so budget is now `sims_per_action * n_legal`, clamped —
     same wall-clock, far more search where branching is actually wide.

  2. Forced positions are skipped entirely. 2.1% of recorded positions had a
     single legal action: a full MCTS search that cannot change the move, and
     a training target that is by construction a one-hot the network gains
     nothing from matching.

  3. Root determinization. `GameState` carries the full remaining deck order,
     so MCTS could read exactly which card refills a pyramid slot after a buy —
     information the observation deliberately hides. The search therefore built
     policy targets the network can never reproduce. Reshuffling the unseen
     decks at the root makes the search plan against a random future instead,
     which is correct in expectation.

  4. Dirichlet alpha scales as `10 / n_legal` (the standard AZ heuristic)
     rather than a flat 0.3 tuned for chess-sized branching.

  5. `temperature_moves` counts phase-decisions, of which a game has ~150, not
     turns. The old default of 15 stopped exploring after roughly three turns.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from splendor_duel.env import encode_state, legal_mask
from splendor_duel.env.action_map import index_to_action
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState

from .mcts_az import NetworkEvaluator, run_mcts, _auto_resolve_trivial
from .self_play import TrainingExample

# A game is ~150 phase-decisions; this only catches pathological loops, so it
# sits far above any real game.  See MAX_GAME_TURNS in self_play.py.
MAX_GAME_DECISIONS = 1000


def determinize(state: GameState, rng: random.Random) -> GameState:
    """Reshuffle the face-down decks so search cannot read the future.

    Every remaining deck card is equally unknown to both players, so permuting
    them yields an equally-plausible world. Cheap: three list shuffles.
    """
    s = state.copy()
    for lvl, deck in s.decks.items():
        cards = list(deck)
        rng.shuffle(cards)
        s.decks[lvl] = cards
    return s


def adaptive_sims(
    n_legal: int, sims_per_action: int, min_sims: int, max_sims: int,
) -> int:
    return int(min(max(sims_per_action * n_legal, min_sims), max_sims))


def _sample_action(visits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature < 1e-6:
        return int(np.argmax(visits))
    probs = visits if temperature == 1.0 else np.power(visits, 1.0 / temperature)
    total = probs.sum()
    if total < 1e-12:
        nonzero = visits > 0
        if not nonzero.any():
            return int(np.argmax(visits))
        probs = nonzero.astype(np.float64)
        total = probs.sum()
    return int(rng.choice(len(probs), p=probs / total))


def generate_game(
    network,
    sims_per_action: int = 6,
    min_sims: int = 32,
    max_sims: int = 320,
    c_puct: float = 1.5,
    temperature_moves: int = 40,
    dirichlet_eps: float = 0.25,
    dirichlet_scale: float = 10.0,
    top_k: int = 0,
    pcr_full_prob: float = 0.0,   # 0 disables PCR entirely
    pcr_fast_sims: int = 64,
    pcr_full_sims: int = 256,
    cards_path: str = "data/cards.json",
    device: str = "cpu",
    rng: Optional[np.random.Generator] = None,
    py_rng: Optional[random.Random] = None,
) -> tuple[list[TrainingExample], Optional[int], int, int]:
    """Play one self-play game.

    Returns (examples, winner, n_decisions, total_sims_spent).
    """
    if rng is None:
        rng = np.random.default_rng()
    if py_rng is None:
        py_rng = random.Random(int(rng.integers(1 << 30)))

    evaluator = NetworkEvaluator(network, device=device)
    state = GameState.new_game(cards_path)
    examples: list[TrainingExample] = []
    decisions = 0
    sims_spent = 0

    while not state.is_game_over and decisions < MAX_GAME_DECISIONS:
        # Resolve forced/trivial phases before deciding anything, so the mask
        # below describes the position MCTS would actually search.
        state = _auto_resolve_trivial(state)
        if state.is_game_over:
            break

        mask = legal_mask(state)
        n_legal = int(mask.sum())
        if n_legal == 0:
            break

        if n_legal == 1:
            # Forced move: no search, no training example.
            state = GameEngine.apply_action(state, index_to_action(int(np.argmax(mask))))
            decisions += 1
            continue

        # Playout cap randomization (Wu 2019). Strength per move saturates far
        # more slowly than cost grows, so paying a deep search on EVERY move
        # buys little play quality while halving how many games the budget
        # affords. Instead: most moves get a cheap search (they still yield a
        # value target, which only needs the game outcome), and a minority get
        # a deep one — and only those become policy targets.
        is_full = pcr_full_prob <= 0.0 or rng.random() < pcr_full_prob
        if pcr_full_prob > 0.0:
            cap = pcr_full_sims if is_full else pcr_fast_sims
            n_sims = adaptive_sims(n_legal, sims_per_action, min_sims, cap)
            # Exploration noise only on the searches we actually learn a
            # policy from; on fast moves it would just add unrecorded variance.
            eps = dirichlet_eps if is_full else 0.0
        else:
            n_sims = adaptive_sims(n_legal, sims_per_action, min_sims, max_sims)
            eps = dirichlet_eps
        search_root = determinize(state, py_rng)

        visits, _root, resolved = run_mcts(
            search_root,
            evaluator,
            n_simulations=n_sims,
            c_puct=c_puct,
            dirichlet_alpha=min(1.0, dirichlet_scale / n_legal),
            dirichlet_eps=eps,
            top_k=top_k,
        )
        sims_spent += n_sims

        total_visits = visits.sum()
        if total_visits <= 0:
            actions = GameEngine.get_legal_actions(state)
            if not actions:
                break
            state = GameEngine.apply_action(state, actions[0])
            decisions += 1
            continue

        # Record against the TRUE state, not the determinized one: the
        # observation hides deck order, so both encode identically, but this
        # keeps the recorded obs unambiguously the real position.
        examples.append(TrainingExample(
            obs=encode_state(state),
            policy=(visits / total_visits).astype(np.float32),
            mask=mask,
            value=0.0,  # backfilled from the outcome below
            current_player=state.current_player,
            policy_weight=1.0 if is_full else 0.0,
        ))

        temperature = 1.0 if decisions < temperature_moves else 0.0
        action_idx = _sample_action(visits, temperature, rng)
        try:
            state = GameEngine.apply_action(state, index_to_action(action_idx))
        except Exception:
            actions = GameEngine.get_legal_actions(state)
            if not actions:
                break
            state = GameEngine.apply_action(state, actions[0])
        decisions += 1

    winner = state.winner if state.is_game_over else None
    for ex in examples:
        ex.value = 0.0 if winner is None else (1.0 if ex.current_player == winner else -1.0)

    return examples, winner, decisions, sims_spent


def generate_batch(
    network,
    n_games: int,
    seed: Optional[int] = None,
    **kwargs,
) -> tuple[list[TrainingExample], dict]:
    """Generate `n_games` self-play games. kwargs forward to generate_game."""
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    all_examples: list[TrainingExample] = []
    p0 = p1 = draws = 0
    total_decisions = 0
    total_sims = 0

    for _ in range(n_games):
        examples, winner, decisions, sims = generate_game(
            network, rng=rng, py_rng=py_rng, **kwargs,
        )
        all_examples.extend(examples)
        total_decisions += decisions
        total_sims += sims
        if winner == 0:
            p0 += 1
        elif winner == 1:
            p1 += 1
        else:
            draws += 1

    return all_examples, {
        "games": n_games,
        "p0_wins": p0,
        "p1_wins": p1,
        "draws": draws,
        "avg_decisions": total_decisions / max(n_games, 1),
        "avg_sims_per_move": total_sims / max(len(all_examples), 1),
        "total_examples": len(all_examples),
    }
