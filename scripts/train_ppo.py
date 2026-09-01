"""
train_ppo.py — Train a PPO agent for Splendor Duel.

Usage:
    # Quick test (5 min, should start beating Random)
    python scripts/train_ppo.py --opponent random --steps 50000

    # Standard training vs Greedy (~1-2 hours)
    python scripts/train_ppo.py --opponent greedy --steps 500000

    # With GPU
    python scripts/train_ppo.py --device cuda --steps 500000

    # Resume from checkpoint
    python scripts/train_ppo.py --resume checkpoints/ppo_100.pt --steps 500000

    # Evaluate a trained model
    python scripts/train_ppo.py --eval checkpoints/ppo_final.pt --eval-games 50
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _parse_hidden_sizes(s: str) -> tuple[int, ...]:
    """'512,512,512,512' -> (512, 512, 512, 512). '256,128' reproduces the
    original (legacy) trunk size."""
    return tuple(int(x) for x in s.split(","))


def train(args):
    from splendor_duel.agents.ppo import PPOTrainer

    trainer = PPOTrainer(
        opponent=args.opponent,
        curriculum=args.curriculum,
        lr=args.lr,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        entropy_coeff=args.entropy_coeff,
        device=args.device,
        cards_path="data/cards.json",
        hidden_sizes=args.hidden_sizes,
    )

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} (update {trainer.total_updates})")

    mode = "curriculum (Random→Greedy)" if args.curriculum else f"vs {args.opponent}"
    print(f"Training PPO {mode}")
    print(f"  Steps: {args.steps}, Device: {args.device}")
    print(f"  LR: {args.lr}, Batch: {args.batch_size}, Steps/update: {args.n_steps}")
    print(f"  Hidden sizes (requested): {args.hidden_sizes} "
          f"— actual network: {trainer.network.hidden_sizes}")
    print()

    trainer.train(
        total_steps=args.steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
    )


def evaluate(args):
    import random
    from splendor_duel.agents.ppo import PPOAgent
    from splendor_duel.agents import RandomAgent, GreedyAgent, play_game

    agent = PPOAgent.load(args.eval, device=args.device)
    print(f"Evaluating {agent.name}")

    opponents = {
        "Random": lambda: RandomAgent(),
        "Greedy": lambda: GreedyAgent(),
    }

    for opp_name, opp_factory in opponents.items():
        wins = 0
        for i in range(args.eval_games):
            random.seed(i)
            if i % 2 == 0:
                r = play_game(agent, opp_factory(), "data/cards.json")
                if r.winner == 0: wins += 1
            else:
                r = play_game(opp_factory(), agent, "data/cards.json")
                if r.winner == 1: wins += 1
        wr = wins / args.eval_games
        print(f"  vs {opp_name}: {wins}/{args.eval_games} ({wr:.0%})")


def main():
    parser = argparse.ArgumentParser(description="Train PPO for Splendor Duel")
    parser.add_argument("--eval", type=str, default=None,
                        help="Evaluate checkpoint instead of training")
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--opponent", default="greedy",
                        choices=["random", "greedy", "self"])
    parser.add_argument("--curriculum", action="store_true",
                        help="Curriculum: start vs Random, gradually shift to Greedy")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-sizes", type=_parse_hidden_sizes, default="512,512,512,512",
                        help="Comma-separated trunk layer widths for a fresh network "
                             "(ignored when --resume loads a checkpoint — its own "
                             "architecture wins). Use '256,128' for the original/legacy size.")
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--save-dir", default="checkpoints")

    args = parser.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()