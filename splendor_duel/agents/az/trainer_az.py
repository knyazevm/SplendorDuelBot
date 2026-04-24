"""
trainer_az.py — AlphaZero training loop.

Cycle:
  1. Generate N self-play games using current network
  2. Add examples to replay buffer (drop oldest if over capacity)
  3. Train K epochs on buffer (sample batches)
  4. Save checkpoint, evaluate, repeat

Loss:
  L = L_policy + L_value
  L_policy = -sum(mcts_policy * log_softmax(logits_masked))
  L_value  = MSE(value_pred, outcome)
"""
from __future__ import annotations

import copy
import random
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from splendor_duel.env import OBS_SIZE, N_ACTIONS
from splendor_duel.agents.ppo.network import SplendorNetwork

from .self_play import TrainingExample, generate_batch


class ReplayBuffer:
    """
    Bounded deque of TrainingExamples. Oldest dropped when full.
    """

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer: deque[TrainingExample] = deque(maxlen=capacity)

    def add_many(self, examples: list[TrainingExample]):
        self.buffer.extend(examples)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, batch_size: int):
        """Randomly sample a batch. Returns stacked numpy arrays."""
        n = len(self.buffer)
        indices = np.random.choice(n, size=min(batch_size, n), replace=False)
        batch = [self.buffer[i] for i in indices]
        obs = np.stack([e.obs for e in batch])
        policy = np.stack([e.policy for e in batch])
        value = np.array([e.value for e in batch], dtype=np.float32)
        mask = np.stack([e.mask for e in batch])
        return obs, policy, value, mask


class AZTrainer:
    """
    AlphaZero trainer.

    Parameters:
        lr:             learning rate
        buffer_capacity: replay buffer size
        games_per_iter:  self-play games per iteration
        epochs_per_iter: training epochs per iteration
        batch_size:      training batch size
        n_simulations:   MCTS sims per move during self-play
        c_puct:          PUCT exploration constant
        temperature_moves: first N moves use T=1 sampling
        dirichlet_eps:   root noise for exploration
        device:          "cpu" or "cuda"
        init_checkpoint: optional PPO checkpoint to warm-start network
    """

    def __init__(
        self,
        lr: float = 1e-3,
        buffer_capacity: int = 50_000,
        games_per_iter: int = 20,
        epochs_per_iter: int = 5,
        batch_size: int = 256,
        n_simulations: int = 50,
        c_puct: float = 1.5,
        temperature_moves: int = 15,
        dirichlet_eps: float = 0.25,
        value_coeff: float = 1.0,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        cards_path: str = "data/cards.json",
        init_checkpoint: Optional[str] = None,
        eval_every: int = 0,
        eval_games: int = 10,
        eval_opponents: Optional[list[str]] = None,
    ):
        self.device = torch.device(device)
        self.cards_path = cards_path
        self.games_per_iter = games_per_iter
        self.epochs_per_iter = epochs_per_iter
        self.batch_size = batch_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature_moves = temperature_moves
        self.dirichlet_eps = dirichlet_eps
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.eval_opponents = eval_opponents or ["greedy"]

        self.network = SplendorNetwork().to(self.device)

        if init_checkpoint:
            self._load_weights_only(init_checkpoint)
            print(f"Initialized network from {init_checkpoint}")

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay,
        )

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.total_iterations = 0
        self.total_games = 0

    def _load_weights_only(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        if "network" in data:
            self.network.load_state_dict(data["network"])
        else:
            self.network.load_state_dict(data)

    def train(
        self,
        total_iterations: int,
        save_dir: str = "checkpoints_az",
        save_interval: int = 5,
        log_interval: int = 1,
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        t_start = time.time()

        for it in range(total_iterations):
            t_iter = time.time()

            # ── Self-play ──────────────────────────────────────
            self.network.eval()
            examples, stats = generate_batch(
                self.network,
                n_games=self.games_per_iter,
                n_simulations=self.n_simulations,
                c_puct=self.c_puct,
                temperature_moves=self.temperature_moves,
                dirichlet_eps=self.dirichlet_eps,
                cards_path=self.cards_path,
                device=str(self.device),
            )
            self.buffer.add_many(examples)
            self.total_games += stats["games"]
            t_selfplay = time.time() - t_iter

            # ── Training ───────────────────────────────────────
            t_train_start = time.time()
            losses = self._train_epochs()
            t_train = time.time() - t_train_start
            self.total_iterations += 1

            if (it + 1) % log_interval == 0:
                elapsed = time.time() - t_start
                print(
                    f"Iter {self.total_iterations:3d} | "
                    f"Games {self.total_games:4d} | "
                    f"Buffer {len(self.buffer):5d} | "
                    f"P0 {stats['p0_wins']:2d}/{stats['p1_wins']:2d}/{stats['draws']:2d} P1/D | "
                    f"AvgT {stats['avg_turns']:5.1f} | "
                    f"PLoss {losses['policy_loss']:.3f} | "
                    f"VLoss {losses['value_loss']:.3f} | "
                    f"SelfPlay {t_selfplay:.0f}s | "
                    f"Train {t_train:.1f}s | "
                    f"Total {elapsed:.0f}s"
                )

            if save_interval > 0 and (it + 1) % save_interval == 0:
                path = f"{save_dir}/az_{self.total_iterations}.pt"
                self.save(path)
                print(f"  → Saved {path}")

            # Periodic evaluation vs reference agents
            if self.eval_every > 0 and (it + 1) % self.eval_every == 0:
                self._run_evaluation()

        self.save(f"{save_dir}/az_final.pt")
        print(f"\nTraining complete: {self.total_iterations} iterations, "
              f"{self.total_games} games in {time.time()-t_start:.0f}s")

    def _train_epochs(self) -> dict:
        """Run epochs_per_iter epochs of SGD over buffer."""
        self.network.train()
        if len(self.buffer) < self.batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        total_p = 0.0
        total_v = 0.0
        n_batches = 0

        # How many batches per epoch (cover buffer ~once)
        batches_per_epoch = max(1, len(self.buffer) // self.batch_size)

        for epoch in range(self.epochs_per_iter):
            for _ in range(batches_per_epoch):
                obs_np, policy_np, value_np, mask_np = self.buffer.sample_batch(self.batch_size)

                obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                target_policy = torch.tensor(policy_np, dtype=torch.float32, device=self.device)
                target_value = torch.tensor(value_np, dtype=torch.float32, device=self.device)
                mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)

                # Forward
                logits, value_pred = self.network(obs)
                value_pred = value_pred.squeeze(-1)

                # Masked log-softmax for policy loss
                logits_m = logits.masked_fill(~mask, -1e9)
                logits_m = logits_m - logits_m.max(dim=-1, keepdim=True).values
                log_probs = logits_m - torch.log(torch.exp(logits_m).sum(dim=-1, keepdim=True) + 1e-10)

                # Policy loss: cross-entropy with MCTS policy
                policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

                # Value loss: MSE
                value_loss = F.mse_loss(value_pred, target_value)

                loss = policy_loss + self.value_coeff * value_loss

                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_p += policy_loss.item()
                total_v += value_loss.item()
                n_batches += 1

        return {
            "policy_loss": total_p / max(n_batches, 1),
            "value_loss": total_v / max(n_batches, 1),
        }

    def _run_evaluation(self):
        """Play eval_games against reference agents, print win-rate."""
        import random as stdlib_random
        from .az_agent import AZAgent
        from splendor_duel.agents import (
            RandomAgent, GreedyAgent, MCTSAgent, play_game,
        )

        # Build agent from current network (low sims for speed during eval)
        eval_agent = AZAgent(
            network=self.network,
            n_simulations=max(50, self.n_simulations),
            c_puct=self.c_puct,
            deterministic=True,
            device=str(self.device),
            name=f"AZ(it{self.total_iterations})",
        )

        opponent_factories = {
            "random": lambda: RandomAgent(),
            "greedy": lambda: GreedyAgent(),
            "mcts": lambda: MCTSAgent(
                iterations=100, rollout='greedy', rollout_depth=20,
            ),
        }

        for opp_name in self.eval_opponents:
            if opp_name not in opponent_factories:
                continue
            wins = 0
            for i in range(self.eval_games):
                stdlib_random.seed(i * 17 + self.total_iterations)
                if i % 2 == 0:
                    r = play_game(
                        eval_agent, opponent_factories[opp_name](),
                        self.cards_path,
                    )
                    if r.winner == 0:
                        wins += 1
                else:
                    r = play_game(
                        opponent_factories[opp_name](), eval_agent,
                        self.cards_path,
                    )
                    if r.winner == 1:
                        wins += 1
            wr = wins / self.eval_games
            print(f"  Eval vs {opp_name}: {wins}/{self.eval_games} ({wr:.0%})")

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_iterations": self.total_iterations,
            "total_games": self.total_games,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(data["network"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        self.total_iterations = data.get("total_iterations", 0)
        self.total_games = data.get("total_games", 0)