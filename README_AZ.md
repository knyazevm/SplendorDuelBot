# AlphaZero Training — Quick Start

## Files

```
splendor_duel/agents/az/
├── __init__.py
├── mcts_az.py        # AlphaZero MCTS (PUCT, neural network eval)
├── self_play.py      # Self-play data generation
├── trainer_az.py     # Training loop + replay buffer
└── az_agent.py       # BaseAgent wrapper for play

scripts/
└── train_az.py       # CLI entry point

splendor_duel/web/server.py  # updated to auto-detect AZ/PPO checkpoints
```

## Step 1: Smoke test (5-10 min)

Run a minimal training cycle to verify everything works:

```bash
python scripts/train_az.py --iterations 2 --games-per-iter 2 --simulations 20
```

Expected: completes without errors, saves `checkpoints_az/az_1.pt` and `az_2.pt`.

## Step 2: Warm-start from PPO (recommended)

Initialize from your best PPO checkpoint to skip the "random policy" phase:

```bash
python scripts/train_az.py \
    --init checkpoints/ppo_100.pt \
    --iterations 50 \
    --games-per-iter 20 \
    --simulations 50
```

Expected duration: ~15-20 hours on CPU. Checkpoints saved every 5 iterations.

## Step 3: Evaluate

Check how the model plays against all baselines:

```bash
python scripts/train_az.py --eval checkpoints_az/az_50.pt --eval-games 20 --sims 200
```

Or against a specific opponent:

```bash
python scripts/train_az.py --eval checkpoints_az/az_final.pt --eval-vs mcts --sims 200
```

## Step 4: Play against AZ in browser

AZ checkpoints are auto-registered in the web UI. Just start the server:

```bash
python scripts/play_web.py
```

Then select e.g. `az_50` from the dropdown. (Note: MCTS-based agents are slow
in the browser — each move takes a few seconds.)

## Key parameters

| Flag | Default | Notes |
|---|---|---|
| `--iterations` | 50 | Total AZ iterations (self-play + train cycles) |
| `--games-per-iter` | 20 | Self-play games per iteration |
| `--simulations` | 50 | MCTS sims per move during self-play |
| `--epochs-per-iter` | 5 | Training epochs per iteration |
| `--batch-size` | 256 | SGD batch size |
| `--buffer-size` | 50000 | Replay buffer capacity |
| `--lr` | 1e-3 | Learning rate |
| `--c-puct` | 1.5 | PUCT exploration constant |
| `--init` | None | Path to PPO checkpoint for warm-start |
| `--resume` | None | Resume from AZ checkpoint |

## Log output

```
Iter   1 | Games   20 | Buffer  4820 | P0 12/ 8/ 0 P1/D | AvgT 121.5 | \
    PLoss 2.341 | VLoss 0.872 | SelfPlay 450s | Train 8s | Total 458s
```

- `Buffer`: examples in replay buffer
- `P0/P1/D`: player 0 wins / player 1 wins / draws
- `AvgT`: average game length
- `PLoss`: policy cross-entropy with MCTS visits
- `VLoss`: value MSE with outcomes
- `SelfPlay`: time spent on self-play
- `Train`: time spent on SGD

## Tuning for speed vs quality

**Speed priority (CPU, quick experiments):**
```bash
--games-per-iter 10 --simulations 30 --epochs-per-iter 3
```
~3-5 min per iteration.

**Quality priority (longer runs):**
```bash
--games-per-iter 30 --simulations 100 --epochs-per-iter 10
```
~30-60 min per iteration but better learning signal.

## What to watch for

**Good signs:**
- `PLoss` decreasing (network learning to mimic MCTS)
- `VLoss` decreasing (value predictions matching outcomes)
- `AvgT` staying in 100-200 range (not hitting 250 limit)
- P0/P1 split roughly balanced (neither player always wins)

**Warning signs:**
- `PLoss` explodes (> 10) → reduce `--lr` to 3e-4 or 1e-4
- All games go to max turns (250) → network isn't learning to end games
- P0 or P1 wins 100% → something wrong with perspective/value sign
- Out of memory → reduce `--batch-size` or `--buffer-size`

## Resume / continue training

```bash
# Continue from last checkpoint
python scripts/train_az.py --resume checkpoints_az/az_25.pt --iterations 50
```

The `--resume` flag loads network weights AND optimizer state, so momentum is preserved.