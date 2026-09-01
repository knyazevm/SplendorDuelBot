"""
self_play.py — Generate training data via AlphaZero self-play.

One network plays both sides. For each position, AZ-MCTS computes
visit counts over actions, and the move is sampled (with temperature)
or chosen greedily. Game outcomes are backfilled into training examples.

Training example:
    obs:     np.float32[OBS_SIZE]
    policy:  np.float32[N_ACTIONS] — visit counts normalised to sum 1
    value:   float in [-1, 1] — outcome from position's active player's POV
    mask:    np.bool[N_ACTIONS]   — legal action mask (for potential masked loss)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from splendor_duel.env import N_ACTIONS, OBS_SIZE, encode_state, legal_mask
from splendor_duel.env.action_map import index_to_action
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState

from .mcts_az import NetworkEvaluator, run_mcts

MAX_GAME_TURNS = 250  # safety


@dataclass
class TrainingExample:
    obs: np.ndarray  # [OBS_SIZE] float32
    policy: np.ndarray  # [N_ACTIONS] float32
    mask: np.ndarray  # [N_ACTIONS] bool
    value: float  # filled after game ends
    current_player: int  # who acted at this position
    # 1.0 = train the policy head on this position, 0.0 = value target only.
    # Playout cap randomization (see self_play_v2) records most positions with
    # weight 0: they were played with a small search whose visit distribution
    # is too noisy to be a policy target, but whose game outcome is still a
    # perfectly good value target. Defaults to 1.0 so every existing caller
    # and every previously pickled buffer behaves exactly as before.
    policy_weight: float = 1.0


def _sample_action(visits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """
    Sample an action from visit counts.

    temperature = 1.0: proportional to visits
    temperature → 0:   argmax (deterministic)
    """
    if temperature < 1e-6:
        return int(np.argmax(visits))

    # visits^(1/T) renormalised
    if temperature == 1.0:
        probs = visits
    else:
        probs = np.power(visits, 1.0 / temperature)

    total = probs.sum()
    if total < 1e-12:
        # Fallback: uniform over non-zero
        nonzero = (visits > 0)
        if not nonzero.any():
            return 0
        probs = nonzero.astype(np.float32)
        total = probs.sum()

    probs = probs / total
    return int(rng.choice(len(probs), p=probs))


def generate_game(
        network,
        n_simulations: int = 100,
        c_puct: float = 1.5,
        temperature_moves: int = 15,
        dirichlet_eps: float = 0.25,
        dirichlet_alpha: float = 0.3,
        cards_path: str = "data/cards.json",
        device: str = "cpu",
        rng: Optional[np.random.Generator] = None,
) -> tuple[list[TrainingExample], Optional[int], int]:
    """
    Play one AZ vs AZ game (same network both sides).

    Args:
        n_simulations:     MCTS simulations per move
        temperature_moves: first N moves use T=1 (sampled), rest T=0 (argmax)
        dirichlet_eps:     exploration noise on root (0 disables, 0.25 typical)

    Returns:
        examples: list of TrainingExample with value filled from outcome
        winner:   0 or 1 (None if draw/truncated)
        turns:    number of turns played
    """
    if rng is None:
        rng = np.random.default_rng()

    evaluator = NetworkEvaluator(network, device=device)

    state = GameState.new_game(cards_path)
    examples: list[TrainingExample] = []

    move_count = 0
    while not state.is_game_over and move_count < MAX_GAME_TURNS:
        # Run MCTS
        visits, root, resolved = run_mcts(
            state, evaluator,
            n_simulations=n_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
        )

        if resolved.is_game_over:
            state = resolved
            break

        # No MCTS decisions (shouldn't happen after _auto_resolve_trivial, but safety)
        if visits.sum() == 0:
            actions = GameEngine.get_legal_actions(resolved)
            if not actions:
                break
            state = GameEngine.apply_action(resolved, actions[0])
            move_count += 1
            continue

        # Record training example at the RESOLVED state (the state MCTS saw)
        obs = encode_state(resolved)
        mask = legal_mask(resolved)
        policy = visits / visits.sum()  # normalised to sum 1
        examples.append(TrainingExample(
            obs=obs,
            policy=policy.astype(np.float32),
            mask=mask,
            value=0.0,  # filled below
            current_player=resolved.current_player,
        ))

        # Sample action
        temperature = 1.0 if move_count < temperature_moves else 0.0
        action_idx = _sample_action(visits, temperature, rng)

        # Apply action
        try:
            game_action = index_to_action(action_idx)
            state = GameEngine.apply_action(resolved, game_action)
        except Exception:
            # Should not happen if MCTS only visited legal actions,
            # but be defensive — fall back to top legal action
            actions = GameEngine.get_legal_actions(resolved)
            if not actions:
                break
            state = GameEngine.apply_action(resolved, actions[0])

        move_count += 1

    # Fill in values based on outcome (from each position's active player's POV)
    winner = state.winner if state.is_game_over else None
    for ex in examples:
        if winner is None:
            ex.value = 0.0
        elif ex.current_player == winner:
            ex.value = 1.0
        else:
            ex.value = -1.0

    return examples, winner, move_count


def generate_batch(
        network,
        n_games: int,
        n_simulations: int = 100,
        c_puct: float = 1.5,
        temperature_moves: int = 15,
        dirichlet_eps: float = 0.25,
        cards_path: str = "data/cards.json",
        device: str = "cpu",
        verbose: bool = False,
        seed: Optional[int] = None,
) -> tuple[list[TrainingExample], dict]:
    """
    Generate a batch of self-play games.

    Returns:
        all_examples: flat list of TrainingExample from all games
        stats: {"games": N, "p0_wins": ..., "p1_wins": ..., "draws": ..., "avg_turns": ...}
    """
    rng = np.random.default_rng(seed)
    all_examples: list[TrainingExample] = []
    p0_wins = p1_wins = draws = 0
    total_turns = 0

    for i in range(n_games):
        examples, winner, turns = generate_game(
            network,
            n_simulations=n_simulations,
            c_puct=c_puct,
            temperature_moves=temperature_moves,
            dirichlet_eps=dirichlet_eps,
            cards_path=cards_path,
            device=device,
            rng=rng,
        )
        all_examples.extend(examples)
        total_turns += turns

        if winner == 0:
            p0_wins += 1
        elif winner == 1:
            p1_wins += 1
        else:
            draws += 1

        if verbose:
            print(f"  Game {i + 1}/{n_games}: winner=P{winner}, {turns} turns, "
                  f"{len(examples)} examples")

    stats = {
        "games": n_games,
        "p0_wins": p0_wins,
        "p1_wins": p1_wins,
        "draws": draws,
        "avg_turns": total_turns / max(n_games, 1),
        "total_examples": len(all_examples),
    }
    return all_examples, stats
