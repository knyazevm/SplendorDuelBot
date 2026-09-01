"""
lr_find.py — fastai-style LR range test for SplendorNetwork.

Ramps the learning rate exponentially over batches sampled from an existing
replay buffer, records the loss at each step, and recommends a starting LR
(steepest-descent point in the smoothed loss curve, per Leslie Smith's LR
range test / fastai's lr_find convention).

Works on any replay_buffer.pkl regardless of which network architecture
generated the self-play data — the buffer just holds (obs, policy, mask,
value) arrays, independent of network shape, so it's fine to reuse a buffer
from an older/smaller run to tune LR for a new architecture.

This mirrors AZTrainer._train_epochs()'s exact loss (masked policy
cross-entropy + value_coeff * value MSE, same grad clipping) so the result
is representative of real training steps.

Usage:
    python scripts/lr_find.py --buffer checkpoints_az_tuned/replay_buffer.pkl \
        --hidden-sizes 512,512,512,512 --value-coeff 1.5
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _parse_hidden_sizes(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from splendor_duel.agents.az.trainer_az import ReplayBuffer
    from splendor_duel.agents.ppo.network import SplendorNetwork

    p = argparse.ArgumentParser(description="LR range test against an existing replay buffer")
    p.add_argument("--buffer", required=True, help="Path to a replay_buffer.pkl")
    p.add_argument("--hidden-sizes", type=_parse_hidden_sizes, default="512,512,512,512")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--value-coeff", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-max", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv-out", default=None, help="Optional path to dump (lr, loss) rows")
    args = p.parse_args()

    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # ReplayBuffer.sample_batch uses the global numpy RNG

    buffer = ReplayBuffer(capacity=10 ** 9)  # capacity irrelevant here, just loading
    buffer.load(args.buffer)
    print(f"Loaded {len(buffer)} examples from {args.buffer}")
    if len(buffer) < args.batch_size:
        raise SystemExit(f"Buffer has fewer examples ({len(buffer)}) than --batch-size ({args.batch_size})")

    network = SplendorNetwork(hidden_sizes=args.hidden_sizes)
    print(f"Network hidden_sizes: {network.hidden_sizes}")
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr_min, weight_decay=args.weight_decay)

    lrs = [args.lr_min * (args.lr_max / args.lr_min) ** (i / (args.steps - 1)) for i in range(args.steps)]

    network.train()
    records: list[tuple[float, float]] = []  # (lr, loss)
    for step, lr in enumerate(lrs):
        for g in optimizer.param_groups:
            g["lr"] = lr

        obs_np, policy_np, value_np, mask_np = buffer.sample_batch(args.batch_size)
        obs = torch.tensor(obs_np, dtype=torch.float32)
        target_policy = torch.tensor(policy_np, dtype=torch.float32)
        target_value = torch.tensor(value_np, dtype=torch.float32)
        mask = torch.tensor(mask_np, dtype=torch.bool)

        logits, value_pred = network(obs)
        value_pred = value_pred.squeeze(-1)

        logits_m = logits.masked_fill(~mask, -1e9)
        logits_m = logits_m - logits_m.max(dim=-1, keepdim=True).values
        log_probs = logits_m - torch.log(torch.exp(logits_m).sum(dim=-1, keepdim=True) + 1e-10)
        policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
        value_loss = F.mse_loss(value_pred, target_value)
        loss = policy_loss + args.value_coeff * value_loss

        if not torch.isfinite(loss):
            print(f"step {step:3d} | lr {lr:.2e} -> non-finite loss, stopping early")
            break

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
        optimizer.step()

        records.append((lr, loss.item()))
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:3d} | lr {lr:.2e} | loss {loss.item():.4f}")

        # Stop early once loss has clearly blown up relative to its best point —
        # no point ramping further, and it keeps the run fast.
        if len(records) > 20:
            best = min(l for _, l in records)
            if loss.item() > 4 * best:
                print(f"step {step:3d} | loss diverging (>4x best), stopping early")
                break

    if args.csv_out:
        with open(args.csv_out, "w") as f:
            f.write("lr,loss\n")
            for lr, loss in records:
                f.write(f"{lr},{loss}\n")
        print(f"Wrote {len(records)} rows to {args.csv_out}")

    # ── Analysis: bin by decade of LR rather than reading raw point-to-point
    # slope — single-batch loss here is noisy enough (different random batch
    # each step) that a 2-point slope just chases noise. A per-decade
    # mean/min is far more robust, and the "mean jumps up, max spikes" decade
    # is a clear, visible divergence signal even with noisy data.
    import collections
    raw_lrs = [l for l, _ in records]
    raw_losses = [x for _, x in records]
    bins: dict[int, list[float]] = collections.defaultdict(list)
    for lr, loss in records:
        bins[math.floor(math.log10(lr))].append(loss)

    print()
    print("Loss by LR decade:")
    stats = {}
    for decade in sorted(bins):
        vals = bins[decade]
        mean, mn, mx = sum(vals) / len(vals), min(vals), max(vals)
        stats[decade] = (mean, mn, mx)
        print(f"  1e{decade:+d} to 1e{decade + 1:+d} | n={len(vals):3d} | "
              f"mean={mean:.3f} | min={mn:.3f} | max={mx:.3f}")

    # Best decade = lowest mean loss among decades that don't show a large
    # max/min spread (a wide spread within a decade means instability is
    # already creeping in, even if the mean still looks OK).
    global_min = min(l for _, l in records)
    candidates = [d for d, (mean, mn, mx) in stats.items() if mx < 2.5 * global_min]
    best_decade = min(candidates, key=lambda d: stats[d][0]) if candidates else min(stats, key=lambda d: stats[d][0])
    # Recommend the geometric middle of the best stable decade.
    recommended = 10 ** (best_decade + 0.5)

    print()
    print(f"Best stable decade: 1e{best_decade:+d} to 1e{best_decade + 1:+d} "
          f"(mean={stats[best_decade][0]:.3f}, max={stats[best_decade][2]:.3f})")
    print(f"Recommended LR (geometric middle of that decade): {recommended:.2e}")
    print("(Read the per-decade table above too — this picks a decade, not a single noisy point;")
    print(" use your judgement on where in that decade to land.)")


if __name__ == "__main__":
    main()
