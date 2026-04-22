"""
trainer.py — PPO training loop for Splendor Duel.

Self-contained: no dependency on stable-baselines.
Designed for masked discrete actions and our Gymnasium env.

Usage:
    trainer = PPOTrainer(opponent="greedy")
    trainer.train(total_games=50000)
    trainer.save("checkpoints/ppo_v1.pt")
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from splendor_duel.env import SplendorDuelEnv, N_ACTIONS, OBS_SIZE
from .network import SplendorNetwork


# ── Rollout buffer ────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """Stores one batch of experience for PPO update."""
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    masks: list = field(default_factory=list)

    def add(self, obs, action, log_prob, value, reward, done, mask):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)

    def clear(self):
        for lst in [self.obs, self.actions, self.log_probs,
                    self.values, self.rewards, self.dones, self.masks]:
            lst.clear()

    def __len__(self):
        return len(self.obs)


def compute_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and returns.

    Returns:
        advantages: np.float32[T]
        returns:    np.float32[T]  (advantages + values = returns)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_done = 0.0 if t == T - 1 else dones[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ── PPO Trainer ───────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    PPO training loop.

    Parameters:
        opponent:        env opponent ("random", "greedy", "self")
        lr:              learning rate
        gamma:           discount factor
        gae_lambda:      GAE λ
        clip_epsilon:    PPO clipping range
        entropy_coeff:   entropy bonus coefficient
        value_coeff:     value loss coefficient
        max_grad_norm:   gradient clipping
        n_steps:         steps to collect before each update
        n_epochs:        SGD epochs per update
        batch_size:      minibatch size for SGD
        device:          "cpu" or "cuda"
    """

    def __init__(
            self,
            opponent: str = "greedy",
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_epsilon: float = 0.2,
            entropy_coeff: float = 0.01,
            value_coeff: float = 0.5,
            max_grad_norm: float = 0.5,
            n_steps: int = 2048,
            n_epochs: int = 4,
            batch_size: int = 256,
            device: str = "cpu",
            cards_path: str = "data/cards.json",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.network = SplendorNetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.env = SplendorDuelEnv(
            opponent=opponent, cards_path=cards_path, max_turns=200,
        )
        self.buffer = RolloutBuffer()

        # Stats
        self.total_games = 0
        self.total_steps = 0
        self.total_updates = 0
        self.win_history: list[float] = []  # rolling window

    def train(
            self,
            total_steps: int = 500_000,
            log_interval: int = 5,
            save_interval: int = 50,
            save_dir: str = "checkpoints",
            eval_games: int = 20,
    ):
        """
        Main training loop.

        Args:
            total_steps:   total environment steps to train
            log_interval:  print stats every N updates
            save_interval: save checkpoint every N updates
            eval_games:    games to play for evaluation at save points
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        t_start = time.time()

        obs_np, info = self.env.reset()
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(info["legal_mask"], dtype=torch.bool, device=self.device)

        steps_done = 0
        ep_rewards = 0.0
        ep_wins = []

        while steps_done < total_steps:
            # ── Collect rollout ───────────────────────────────
            self.buffer.clear()
            self.network.eval()

            for _ in range(self.n_steps):
                action, log_prob, value = self.network.get_action(obs_t, mask_t)
                obs_np, reward, done, truncated, info = self.env.step(action)
                ep_rewards += reward

                self.buffer.add(
                    obs=obs_t.cpu().numpy(),
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=float(done or truncated),
                    mask=mask_t.cpu().numpy(),
                )
                steps_done += 1

                if done or truncated:
                    self.total_games += 1
                    win = 1.0 if reward > 0 else 0.0
                    ep_wins.append(win)
                    ep_rewards = 0.0

                    obs_np, info = self.env.reset()

                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                mask_t = torch.tensor(info["legal_mask"], dtype=torch.bool, device=self.device)

            # ── Compute advantages ────────────────────────────
            with torch.no_grad():
                _, last_value = self.network.forward(obs_t.unsqueeze(0))
                last_value = last_value.item()

            advantages, returns = compute_gae(
                rewards=np.array(self.buffer.rewards, dtype=np.float32),
                values=np.array(self.buffer.values, dtype=np.float32),
                dones=np.array(self.buffer.dones, dtype=np.float32),
                last_value=last_value,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )

            # ── PPO update ────────────────────────────────────
            stats = self._update(advantages, returns)
            self.total_updates += 1

            # ── Logging ───────────────────────────────────────
            if self.total_updates % log_interval == 0:
                recent_wr = np.mean(ep_wins[-100:]) if ep_wins else 0
                elapsed = time.time() - t_start
                sps = steps_done / elapsed if elapsed > 0 else 0
                print(
                    f"Update {self.total_updates:4d} | "
                    f"Steps {steps_done:7d}/{total_steps} | "
                    f"Games {self.total_games:5d} | "
                    f"WinRate {recent_wr:.1%} | "
                    f"Loss {stats['total_loss']:.3f} | "
                    f"Entropy {stats['entropy']:.3f} | "
                    f"SPS {sps:.0f}"
                )

            # ── Save ──────────────────────────────────────────
            if save_interval > 0 and self.total_updates % save_interval == 0:
                path = f"{save_dir}/ppo_{self.total_updates}.pt"
                self.save(path)
                print(f"  → Saved {path}")

        # Final save
        self.save(f"{save_dir}/ppo_final.pt")
        elapsed = time.time() - t_start
        print(f"\nTraining complete: {self.total_games} games, "
              f"{steps_done} steps in {elapsed:.0f}s")

    def _update(self, advantages: np.ndarray, returns: np.ndarray) -> dict:
        """Run PPO SGD epochs on the buffer."""
        self.network.train()
        n = len(self.buffer)

        # Convert buffer to tensors
        b_obs = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32, device=self.device)
        b_actions = torch.tensor(self.buffer.actions, dtype=torch.int64, device=self.device)
        b_old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        b_masks = torch.tensor(np.array(self.buffer.masks), dtype=torch.bool, device=self.device)
        b_advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalise advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_batches = 0

        for _ in range(self.n_epochs):
            # Shuffle indices
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                log_probs, values, entropy = self.network.evaluate_actions(
                    b_obs[idx], b_actions[idx], b_masks[idx],
                )

                # PPO clipped policy loss
                ratio = torch.exp(log_probs - b_old_log_probs[idx])
                adv = b_advantages[idx]
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = nn.functional.mse_loss(values, b_returns[idx])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.value_coeff * v_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.mean().item()
                n_batches += 1

        return {
            "total_loss": (total_pg_loss + total_v_loss) / max(n_batches, 1),
            "policy_loss": total_pg_loss / max(n_batches, 1),
            "value_loss": total_v_loss / max(n_batches, 1),
            "entropy": total_entropy / max(n_batches, 1),
        }

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_games": self.total_games,
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(data["network"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.total_games = data.get("total_games", 0)
        self.total_steps = data.get("total_steps", 0)
        self.total_updates = data.get("total_updates", 0)
