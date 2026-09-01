"""
train_az_parallel.py — Parallel self-play training for AlphaZero.

Uses multiprocessing to run N worker processes generating self-play games
concurrently. Main process aggregates examples, trains network, broadcasts
updated weights.

Speedup: ~N_workers × base speed (on multi-core CPU).
Recommended workers: cpu_count() - 1 (leave one for training + OS).

Usage:
    # Standard run with 4 parallel workers
    python scripts/train_az_parallel.py --workers 4 --iterations 50 --games-per-iter 20

    # Warm-start from PPO
    python scripts/train_az_parallel.py --workers 4 --init checkpoints/ppo_100.pt \
        --iterations 50 --games-per-iter 20 --simulations 50

    # Maximum throughput (16-core RunPod)
    python scripts/train_az_parallel.py --workers 14 --iterations 100 \
        --games-per-iter 40 --simulations 80

Notes:
    - Windows requires spawn method (handled automatically).
    - Each worker uses CPU for inference; don't mix with --device cuda here.
    - Worker processes re-import the module, so keep main() clean.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Allow running as script from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _parse_hidden_sizes(s: str) -> tuple[int, ...]:
    """'512,512,512,512' -> (512, 512, 512, 512). '256,128' reproduces the
    original (legacy) trunk size."""
    return tuple(int(x) for x in s.split(","))


# ── Worker function (top-level for pickling) ──────────────────────────────────

def _worker_run_games(args_tuple):
    """
    Worker process: load network from checkpoint, play N games, return examples.

    Args:
        args_tuple: (checkpoint_path, n_games, n_sims, c_puct, temp_moves,
                     dirichlet_eps, cards_path, seed)
    """
    import numpy as np
    import torch

    (checkpoint_path, n_games, n_sims, c_puct, temp_moves,
     dirichlet_eps, cards_path, seed) = args_tuple

    # Seed this worker uniquely
    np.random.seed(seed)
    import random as stdlib_random
    stdlib_random.seed(seed)
    torch.manual_seed(seed)

    from splendor_duel.agents.ppo.network import load_network_from_checkpoint
    from splendor_duel.agents.az.self_play import generate_batch

    # Load network (sized to match whatever architecture is in the checkpoint)
    network, _ = load_network_from_checkpoint(checkpoint_path, device="cpu")
    network.eval()

    # Generate games
    examples, stats = generate_batch(
        network,
        n_games=n_games,
        n_simulations=n_sims,
        c_puct=c_puct,
        temperature_moves=temp_moves,
        dirichlet_eps=dirichlet_eps,
        cards_path=cards_path,
        device="cpu",
        seed=seed,
    )

    # Convert TrainingExamples to tuples for lightweight IPC
    # (pickling TrainingExample dataclass works but we serialise arrays directly)
    serialised = [
        (ex.obs, ex.policy, ex.mask, ex.value, ex.current_player)
        for ex in examples
    ]
    return serialised, stats


def _tuples_to_examples(tuples):
    """Convert serialised tuples back to TrainingExample."""
    from splendor_duel.agents.az.self_play import TrainingExample
    return [
        TrainingExample(obs=t[0], policy=t[1], mask=t[2], value=t[3], current_player=t[4])
        for t in tuples
    ]


# ── Main training loop ────────────────────────────────────────────────────────

def train_parallel(args):
    # Import heavy deps after fork protection
    import torch
    import numpy as np
    from splendor_duel.agents.az import AZTrainer

    # Set up trainer but skip its internal self-play by overriding after construct
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
        device="cpu",  # training happens here, but workers are CPU too
        cards_path="data/cards.json",
        init_checkpoint=args.init,
        eval_every=args.eval_every,
        eval_games=args.eval_games_during_train,
        eval_opponents=args.eval_vs_during_train.split(",") if args.eval_vs_during_train else ["greedy"],
        hidden_sizes=args.hidden_sizes,
    )

    if args.resume:
        trainer.load(args.resume)
        # load() restores the optimizer's state_dict, which includes the lr
        # that was active when the checkpoint was saved — override it so
        # --lr actually takes effect on resume.
        for group in trainer.optimizer.param_groups:
            group["lr"] = args.lr
        buffer_restored = trainer.load_buffer_near(args.resume)
        buffer_msg = (f"buffer restored ({len(trainer.buffer)} examples)" if buffer_restored
                      else "no replay_buffer.pkl found next to it — starting with empty buffer")
        print(f"Resumed from {args.resume} (iter {trainer.total_iterations}), lr set to {args.lr}, {buffer_msg}")

    print(f"Training AlphaZero (parallel, {args.workers} workers)")
    print(f"  Iterations: {args.iterations}")
    print(f"  Total games/iter: {args.games_per_iter} (~{args.games_per_iter // args.workers}/worker)")
    print(f"  MCTS sims/move: {args.simulations}")
    print(f"  Epochs/iter: {args.epochs_per_iter}")
    print(f"  Buffer: {args.buffer_size}, Batch: {args.batch_size}")
    print(f"  Hidden sizes (requested): {args.hidden_sizes} "
          f"— actual network: {trainer.network.hidden_sizes}")
    if args.init:
        print(f"  Warm-start: {args.init}")
    print()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Distribute games across workers
    games_per_worker = max(1, args.games_per_iter // args.workers)
    remainder = args.games_per_iter - games_per_worker * args.workers

    # Temporary checkpoint for workers to load
    temp_ckpt = f"{args.save_dir}/_latest_worker.pt"
    trainer.save(temp_ckpt)  # initial checkpoint for workers

    # Start worker pool
    mp_ctx = mp.get_context("spawn")
    pool = mp_ctx.Pool(processes=args.workers)

    try:
        for it in range(args.iterations):
            t_iter = time.time()

            # Save current weights for workers to pick up
            trainer.save(temp_ckpt)

            # Build per-worker arg tuples
            worker_args = []
            base_seed = (it + 1) * 1000
            for w in range(args.workers):
                n_games = games_per_worker + (1 if w < remainder else 0)
                worker_args.append((
                    temp_ckpt,
                    n_games,
                    args.simulations,
                    args.c_puct,
                    args.temperature_moves,
                    args.dirichlet_eps,
                    "data/cards.json",
                    base_seed + w,
                ))

            # Dispatch to workers
            t_sp_start = time.time()
            results = pool.map(_worker_run_games, worker_args)
            t_selfplay = time.time() - t_sp_start

            # Aggregate
            all_examples = []
            total_p0 = total_p1 = total_draws = 0
            total_turns = 0
            total_games_this_iter = 0
            for tuples, stats in results:
                all_examples.extend(_tuples_to_examples(tuples))
                total_p0 += stats["p0_wins"]
                total_p1 += stats["p1_wins"]
                total_draws += stats["draws"]
                total_turns += stats["avg_turns"] * stats["games"]
                total_games_this_iter += stats["games"]

            avg_turns = total_turns / max(total_games_this_iter, 1)

            trainer.buffer.add_many(all_examples)
            trainer.total_games += total_games_this_iter

            # Train
            t_train_start = time.time()
            losses = trainer._train_epochs()
            t_train = time.time() - t_train_start
            trainer.total_iterations += 1

            # Log
            elapsed = time.time() - t_start
            print(
                f"Iter {trainer.total_iterations:3d} | "
                f"Games {trainer.total_games:4d} | "
                f"Buffer {len(trainer.buffer):5d} | "
                f"P0 {total_p0:2d}/{total_p1:2d}/{total_draws:2d} P1/D | "
                f"AvgT {avg_turns:5.1f} | "
                f"PLoss {losses['policy_loss']:.3f} | "
                f"VLoss {losses['value_loss']:.3f} | "
                f"SelfPlay {t_selfplay:.0f}s | "
                f"Train {t_train:.1f}s | "
                f"Total {elapsed:.0f}s"
            )

            # Save
            if args.save_interval > 0 and (it + 1) % args.save_interval == 0:
                path = f"{args.save_dir}/az_{trainer.total_iterations}.pt"
                trainer.save(path)
                trainer.buffer.save(f"{args.save_dir}/replay_buffer.pkl")
                print(f"  → Saved {path} (+ buffer, {len(trainer.buffer)} examples)")

            # Evaluate
            if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
                trainer._run_evaluation()

        trainer.save(f"{args.save_dir}/az_final.pt")
        trainer.buffer.save(f"{args.save_dir}/replay_buffer.pkl")
        print(f"\nTraining complete: {trainer.total_iterations} iterations, "
              f"{trainer.total_games} games in {time.time()-t_start:.0f}s")
    finally:
        pool.close()
        pool.join()
        try:
            os.remove(temp_ckpt)
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser(description="Parallel AlphaZero trainer")

    # Parallelism
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                   help="Number of parallel self-play workers")

    # Training
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games-per-iter", type=int, default=20,
                   help="Total games per iter, split across workers")
    p.add_argument("--epochs-per-iter", type=int, default=5)
    p.add_argument("--simulations", type=int, default=50)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init", type=str, default=None)

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
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--save-dir", default="checkpoints_az")

    # Periodic eval
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-games-during-train", type=int, default=10)
    p.add_argument("--eval-vs-during-train", default="greedy")

    args = p.parse_args()
    train_parallel(args)


if __name__ == "__main__":
    # Windows requires this guard
    mp.freeze_support()
    main()