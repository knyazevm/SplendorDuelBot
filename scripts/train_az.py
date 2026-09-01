"""
train_az.py — Train an AlphaZero agent for Splendor Duel.

Usage:
    # Quick smoke test (1 iteration, 2 games, ~10 min)
    python scripts/train_az.py --iterations 1 --games-per-iter 2 --simulations 30

    # Warm-started from PPO checkpoint (recommended)
    python scripts/train_az.py --init checkpoints/ppo_100.pt --iterations 50 \
        --games-per-iter 20 --simulations 50

    # Standard run (~days)
    python scripts/train_az.py --iterations 100 --games-per-iter 30 --simulations 100

    # Resume from AZ checkpoint
    python scripts/train_az.py --resume checkpoints_az/az_25.pt --iterations 100

    # Evaluate a trained model
    python scripts/train_az.py --eval checkpoints_az/az_final.pt --eval-games 20 --sims 200

    # Evaluate specifying opponent
    python scripts/train_az.py --eval checkpoints_az/az_final.pt --eval-vs mcts --sims 200
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _parse_hidden_sizes(s: str) -> tuple[int, ...]:
    """'512,512,512,512' -> (512, 512, 512, 512). '256,128' reproduces the
    original (legacy) trunk size."""
    return tuple(int(x) for x in s.split(","))


def train(args):
    from splendor_duel.agents.az import AZTrainer

    trainer = AZTrainer(
        lr=args.lr,
        buffer_capacity=args.buffer_size,
        games_per_iter=args.games_per_iter,
        epochs_per_iter=args.epochs_per_iter,
        batch_size=args.batch_size,
        n_simulations=args.simulations,
        c_puct=args.c_puct,
        temperature_moves=args.temperature_moves,
        dirichlet_eps=args.dirichlet_eps,
        value_coeff=args.value_coeff,
        device=args.device,
        cards_path="data/cards.json",
        init_checkpoint=args.init,
        eval_every=args.eval_every,
        eval_games=args.eval_games_during_train,
        eval_opponents=args.eval_vs_during_train.split(",") if args.eval_vs_during_train else ["greedy"],
        hidden_sizes=args.hidden_sizes,
    )

    if args.resume:
        trainer.load(args.resume)
        buffer_restored = trainer.load_buffer_near(args.resume)
        buffer_msg = (f"buffer restored ({len(trainer.buffer)} examples)" if buffer_restored
                      else "no replay_buffer.pkl found next to it — starting with empty buffer")
        print(f"Resumed from {args.resume} (iter {trainer.total_iterations}), {buffer_msg}")

    print(f"Training AlphaZero")
    print(f"  Iterations: {args.iterations}")
    print(f"  Games per iter: {args.games_per_iter}")
    print(f"  MCTS sims per move: {args.simulations}")
    print(f"  Epochs per iter: {args.epochs_per_iter}")
    print(f"  Buffer: {args.buffer_size}, Batch: {args.batch_size}")
    print(f"  Hidden sizes (requested): {args.hidden_sizes} "
          f"— actual network: {trainer.network.hidden_sizes}")
    print(f"  Device: {args.device}")
    if args.init:
        print(f"  Warm-start: {args.init}")
    print()

    trainer.train(
        total_iterations=args.iterations,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )


def evaluate(args):
    from splendor_duel.agents.az import AZAgent
    from splendor_duel.agents import (
        RandomAgent, GreedyAgent, MCTSAgent, play_game,
    )
    from splendor_duel.agents import (
        GreedyByChatGPT, GeminiGreedyAgent, GreedyByClaude,
        GreedyByChatGPTV2, GeminiGreedyAgentV2, GreedyByClaudeV2,
    )

    agent = AZAgent.load(
        args.eval, n_simulations=args.sims, device=args.device,
    )
    print(f"Evaluating {agent.name}")

    if args.eval_vs == "all":
        opponents = {
            "Random": lambda: RandomAgent(),
            "Greedy": lambda: GreedyAgent(),
            # "GreedyByChatGPT": lambda: GreedyByChatGPT(seed=1),
            # "GeminiGreedyAgent": lambda: GeminiGreedyAgent(seed=1),
            # "GreedyByClaude": lambda: GreedyByClaude(seed=1),
            # "GreedyByChatGPTV2": lambda: GreedyByChatGPTV2(seed=1),
            # "GeminiGreedyAgentV2": lambda: GeminiGreedyAgentV2(seed=1),
            # "GreedyByClaudeV2": lambda: GreedyByClaudeV2(seed=1),
            "MCTS(100,greedy)": lambda: MCTSAgent(
                iterations=100, rollout='greedy', rollout_depth=20,
            ),
        }
    elif args.eval_vs == "random":
        opponents = {"Random": lambda: RandomAgent()}
    elif args.eval_vs == "greedy":
        opponents = {"Greedy": lambda: GreedyAgent()}
    elif args.eval_vs == "mcts":
        opponents = {"MCTS(100,greedy)": lambda: MCTSAgent(
            iterations=100, rollout='greedy', rollout_depth=20,
        )}
    else:
        raise ValueError(f"Unknown eval-vs: {args.eval_vs}")

    for opp_name, opp_factory in opponents.items():
        wins = 0
        for i in range(args.eval_games):
            random.seed(i * 17 + 1)
            if i % 2 == 0:
                r = play_game(agent, opp_factory(), "data/cards.json")
                if r.winner == 0:
                    wins += 1
            else:
                r = play_game(opp_factory(), agent, "data/cards.json")
                if r.winner == 1:
                    wins += 1
        wr = wins / args.eval_games
        print(f"  vs {opp_name}: {wins}/{args.eval_games} ({wr:.0%})")


def main():
    p = argparse.ArgumentParser(description="Train AlphaZero for Splendor Duel")

    # Mode
    p.add_argument("--eval", type=str, default=None,
                   help="Evaluate checkpoint instead of training")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-vs", default="all",
                   choices=["all", "random", "greedy", "mcts"])
    p.add_argument("--sims", type=int, default=200,
                   help="MCTS sims during eval")

    # Training
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games-per-iter", type=int, default=20)
    p.add_argument("--epochs-per-iter", type=int, default=5)
    p.add_argument("--simulations", type=int, default=50,
                   help="MCTS sims per move during self-play")
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from AZ checkpoint")
    p.add_argument("--init", type=str, default=None,
                   help="Initialize weights from PPO checkpoint")

    # Hyperparameters
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temperature-moves", type=int, default=15)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--value-coeff", type=float, default=1.0)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-sizes", type=_parse_hidden_sizes, default="512,512,512,512",
                   help="Comma-separated trunk layer widths for a fresh network "
                        "(ignored when --resume/--init load a checkpoint — its own "
                        "architecture wins). Use '256,128' for the original/legacy size.")

    # Logging
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--save-dir", default="checkpoints_az")

    # Periodic evaluation during training
    p.add_argument("--eval-every", type=int, default=0,
                   help="Run benchmark vs reference agents every N iters (0=off)")
    p.add_argument("--eval-games-during-train", type=int, default=10,
                   help="Games per benchmark run")
    p.add_argument("--eval-vs-during-train", default="greedy",
                   help="Comma-separated opponents: random, greedy, mcts")

    args = p.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
