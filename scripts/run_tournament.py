"""
run_tournament.py — Round-robin tournament between agents.

Usage:
    python scripts/run_tournament.py                    # quick (10 games each)
    python scripts/run_tournament.py --games 100        # thorough
    python scripts/run_tournament.py --include-mcts     # add MCTS (slower)

Results include:
- Head-to-head win rates
- Average game length
- Victory type distribution
- Confidence intervals (Wilson score)
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, '.')

from splendor_duel.agents import (
    RandomAgent, GreedyAgent, GreedyByChatGPT, GeminiGreedyAgent, GreedyByClaude, MCTSAgent, play_game, GameResult,
)

CARDS_PATH = "data/cards.json"


def run_matchup(
        agent_a_factory, agent_b_factory,
        n_games: int,
        cards_path: str,
) -> list[GameResult]:
    """
    Play n_games between two agent factories.
    Alternates who goes first each game to reduce first-player advantage.
    """
    results = []
    for i in range(n_games):
        a = agent_a_factory()
        b = agent_b_factory()
        # Alternate starting player by swapping agent assignment
        if i % 2 == 0:
            result = play_game(a, b, cards_path)
        else:
            result = play_game(b, a, cards_path)
            # Remap winner index back to original agent perspective
            result = GameResult(
                winner=1 - result.winner,
                victory_type=result.victory_type,
                turns=result.turns,
                steps=result.steps,
                scores=(result.scores[1], result.scores[0]),
                crowns=(result.crowns[1], result.crowns[0]),
                cards=(result.cards[1], result.cards[0]),
                elapsed=result.elapsed,
                agent_names=(result.agent_names[1], result.agent_names[0]),
            )
        results.append(result)
    return results


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def print_results(
        name_a: str, name_b: str, results: list[GameResult]
) -> None:
    n = len(results)
    a_wins = sum(1 for r in results if r.winner == 0)
    b_wins = n - a_wins
    wr = a_wins / n if n > 0 else 0
    lo, hi = wilson_ci(a_wins, n)

    avg_turns = sum(r.turns for r in results) / n
    avg_time = sum(r.elapsed for r in results) / n

    vic_types: dict[str, int] = defaultdict(int)
    for r in results:
        vic_types[r.victory_type] += 1

    print(f"\n{'=' * 60}")
    print(f"  {name_a}  vs  {name_b}  ({n} games)")
    print(f"{'=' * 60}")
    print(f"  {name_a} wins: {a_wins:3d} ({wr:.1%})")
    print(f"  {name_b} wins: {b_wins:3d} ({1 - wr:.1%})")
    print(f"  95% CI for {name_a}: [{lo:.1%}, {hi:.1%}]")
    print(f"  Avg turns: {avg_turns:.1f}  |  Avg time: {avg_time:.2f}s")
    print(f"  Victory types: {dict(vic_types)}")


def main():
    parser = argparse.ArgumentParser(description="Splendor Duel Agent Tournament")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per matchup (default: 20)")
    parser.add_argument("--include-mcts", action="store_true",
                        help="Include MCTS agents")
    parser.add_argument("--mcts-iters", type=int, default=200,
                        help="MCTS iterations (default: 200)")
    parser.add_argument("--mcts-modes", type=str, default="none,greedy,random",
                        help="Comma-separated rollout modes (default: none,greedy,random)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Agent factories (create fresh agent per game)
    agents = {
        "Random": lambda: RandomAgent(),
        "Greedy": lambda: GreedyAgent(),
        # "MCTS_fast": lambda: MCTSAgent(iterations=200, rollout='none'),
        # "MCTS_greedy": lambda: MCTSAgent(iterations=100, rollout='greedy', rollout_depth=20),
        "GreedyByChatGPT": lambda: GreedyByChatGPT(seed=1),
        "GeminiGreedyAgent": lambda: GeminiGreedyAgent(seed=1),
        "GreedyByClaude": lambda: GreedyByClaude(seed=1),
    }
    if args.include_mcts:
        iters = args.mcts_iters
        for mode in args.mcts_modes.split(','):
            mode = mode.strip()
            tag = f"MCTS({iters},{mode})"
            # Capture mode in closure
            agents[tag] = (lambda m: lambda: MCTSAgent(
                iterations=iters, rollout=m, prioritize=True
            ))(mode)

    names = list(agents.keys())
    all_results: dict[tuple[str, str], list[GameResult]] = {}

    print(f"Tournament: {', '.join(names)}")
    print(f"Games per matchup: {args.games}")
    print(f"Cards: {CARDS_PATH}")

    t_start = time.time()

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            print(f"\nPlaying {name_a} vs {name_b}...", end="", flush=True)
            results = run_matchup(
                agents[name_a], agents[name_b],
                args.games, CARDS_PATH,
            )
            all_results[(name_a, name_b)] = results
            a_wins = sum(1 for r in results if r.winner == 0)
            print(f" done ({a_wins}/{args.games})")

    t_total = time.time() - t_start

    # Print all results
    for (name_a, name_b), results in all_results.items():
        print_results(name_a, name_b, results)

    print(f"\nTotal time: {t_total:.1f}s")

    # Summary table
    print(f"\n{'─' * 50}")
    print("SUMMARY (win rates for row agent vs column agent):")
    print(f"{'':>12}", end="")
    for n in names:
        print(f"{n:>12}", end="")
    print()
    for na in names:
        print(f"{na:>12}", end="")
        for nb in names:
            if na == nb:
                print(f"{'—':>12}", end="")
            elif (na, nb) in all_results:
                res = all_results[(na, nb)]
                wr = sum(1 for r in res if r.winner == 0) / len(res)
                print(f"{wr:>11.0%} ", end="")
            elif (nb, na) in all_results:
                res = all_results[(nb, na)]
                wr = sum(1 for r in res if r.winner == 1) / len(res)
                print(f"{wr:>11.0%} ", end="")
            else:
                print(f"{'?':>12}", end="")
        print()


if __name__ == "__main__":
    main()
