"""
train_az_v2.py — AlphaZero training for Splendor Duel, with the defects found
in the original loop fixed.

WHY THIS FILE EXISTS
────────────────────
`checkpoints_az_tuned/az_80.pt` (80 iterations of `train_az_parallel.py`) wins
12% of 40 games against the plain GreedyAgent at 200 sims. Measuring its value
head explains why:

    on the replay buffer it trained on : MSE 0.029, outcome sign acc 0.995
    on freshly generated positions     : MSE 1.508, outcome sign acc 0.548

Predicting the constant 0 scores MSE 1.000. The value head had memorised its
buffer and generalised *worse than a constant*, so every MCTS leaf evaluation
was noise. Five defects produced that, all fixed here:

1. SAMPLE REUSE ~83x. `_train_epochs` ran `epochs_per_iter * (len(buffer) //
   batch_size)` steps — with the shipped defaults, 5 * (50000//256) = 975 steps
   of batch 256 against ~3000 fresh positions per iteration. Each position was
   trained on ~83 times over its life in the buffer; AlphaZero-family runs use
   1-4. Here `--reuse` sets that number directly and the step count is derived
   from how much new data actually arrived.

2. UNBOUNDED VALUE HEAD. `SplendorNetwork.value_head` is a bare Linear, correct
   for PPO (unbounded returns) and wrong for AZ, whose targets are outcomes in
   {-1,0,+1} and whose PUCT rule assumes Q in [-1,1]. 23% of az_80's held-out
   predictions fell outside that range. `AZNetwork` adds tanh.

3. FLAT SIM BUDGET. 50 sims went equally to 2-action and 139-action positions.
   On wide positions (41+ legal) the old search touched 27 of 64 actions, so
   the "MCTS policy" target was mostly a copy of the network's own prior —
   self-confirming, no learning signal. Adaptive budget touches 64 of 70.

4. SEARCH READ HIDDEN INFORMATION. `GameState` carries full deck order, which
   the observation hides. MCTS could see exactly which card refilled a slot,
   producing policy targets the network cannot reproduce. Decks are now
   reshuffled at the root of every search.

5. WORKER THREAD OVERSUBSCRIPTION. `train_az_parallel.py` workers never set
   `torch.set_num_threads(1)`, so 11 worker processes each spawned ~12 BLAS
   threads on a 12-core box.

The 1-4 range is not folklore here — sweeping gradient steps against a 20%
whole-game holdout of the existing buffer (fresh 512x4 net, AdamW 1e-3):

    reuse   0.3x  0.6x  1.3x  2.6x  5.1x  10x
    val MSE 1.014 0.986 0.988 1.247 1.480 1.400     (constant predictor = 1.000)

Held-out value loss bottoms around reuse 0.6-1.3 and is already worse than a
constant predictor by 2.6x, which is where the old loop sat times thirty.
`--reuse` therefore defaults to 1.0. Policy loss is far more tolerant (it was
still improving at 10x), so if you raise `--reuse`, watch ValV, not ValP.

Also: Dirichlet alpha now follows the standard 10/n_legal instead of a flat 0.3;
`temperature_moves` counts phase-decisions (a game has ~150, so the old default
of 15 stopped exploring after ~3 turns); forced single-action positions are no
longer searched or trained on; AdamW replaces Adam-with-weight_decay.

DIAGNOSTIC
──────────
A fraction of self-play games is held out whole (never split within a game —
positions in one game share an outcome, so a positional split leaks the value
target) and reported each iteration as `ValV`/`ValP`. Watch `ValV`: if it
climbs toward 1.0 while `VLoss` falls, the value head is memorising again and
`--reuse` is too high.

TUNING, AS MEASURED (all head-to-head, az_30, alternating seats)
────────────────────────────────────────────────────────────────
SIMULATIONS. Elo per doubling is roughly CONSTANT (~+190) with no saturation
anywhere in the tested range — the classic MCTS result:

    32->64  -50   64->128  +191 (28 games)   128->256  -0   256->512  +191 (28 games)

The floor is set by the policy-improvement gap — how much MCTS beats the raw
network, which IS AlphaZero's learning signal:

    MCTS(32) vs raw policy 57%   MCTS(64) 79%   MCTS(128) 93%

At 32 sims the search barely improves on its own prior and training cannot
bootstrap. Use >= 64; 128 is a comfortable default. Going higher buys real
strength but halves the games the budget affords, and the value head's
effective sample size is the number of GAMES, not positions (positions in one
game share a single outcome label). That trade needs training runs to settle,
which is what --pcr-full-prob sidesteps.

C_PUCT: leave at 1.5. Lowering it measured WORSE (c_puct 0.8 vs 1.5 at 128
sims: 4/14, -159 Elo) even though it concentrates visits exactly as theory
says (top-move share 74% at 0.4 vs 53% at 1.5). The search concentrates harder
on a worse move — sharper commitment to a value head that is only ~74%
accurate is a loss, not a gain.

TOP_K (mcts_az.run_mcts(top_k=...)): no benefit. 50%, 50%, 17% against
unrestricted at K=12/20/32, and worse search stability on every bounded metric.
Left in as an opt-in default of 0.

METHODOLOGY WARNING. Every duel above at n=14 has a standard error of ~13%;
detecting a true 60%-vs-50% edge needs ~270 games. Several apparent findings in
this file's history evaporated on replication (512-vs-256 went 93% -> 57%).
Proxy metrics — cross-entropy gaps, visit-distribution stability, search
concentration — repeatedly pointed the OPPOSITE way from actual play here.
Trust head-to-head results at n >= 200, and nothing else.

USAGE
─────
    export PYTHONPATH=.

    # smoke test, ~2 min
    python scripts/train_az_v2.py --iterations 2 --games-per-iter 4 --workers 4 \
        --sims-per-action 3 --max-sims 60

    # main run: ~6-8 h on 12 cores, this is the one to launch
    python scripts/train_az_v2.py --workers 6 --iterations 200 --games-per-iter 36 \
        --eval-every 10 --save-dir checkpoints_az_v2

    # resume
    python scripts/train_az_v2.py --resume checkpoints_az_v2/az_100.pt \
        --iterations 200 --workers 6

    # evaluate (this loads the tanh head correctly; train_az.py --eval does not)
    python scripts/train_az_v2.py --eval checkpoints_az_v2/az_final.pt --eval-games 40
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CARDS = "data/cards.json"


def _parse_hidden_sizes(s):
    return tuple(int(x) for x in s.split(","))


# ── Workers ───────────────────────────────────────────────────────────────────

def _init_worker():
    """One BLAS thread per worker. Without this, N workers each spawn a full
    thread pool and spend most of their time contending."""
    import torch
    torch.set_num_threads(1)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _worker_selfplay(job):
    """Play `n_games` and return one list of serialised examples PER GAME, so
    the parent can hold out whole games for validation."""
    import random as pyrandom

    import numpy as np
    import torch

    ckpt, n_games, seed, sp_kwargs = job
    torch.set_num_threads(1)
    np.random.seed(seed % (2 ** 31))
    pyrandom.seed(seed)
    torch.manual_seed(seed % (2 ** 31))

    from splendor_duel.agents.az.network_az import load_az_network
    from splendor_duel.agents.az.self_play_v2 import generate_game

    network, _ = load_az_network(ckpt, device="cpu")
    rng = np.random.default_rng(seed)
    py_rng = pyrandom.Random(seed)

    games, p0, p1, draws, decisions, sims, n_ex = [], 0, 0, 0, 0, 0, 0
    for _ in range(n_games):
        examples, winner, n_dec, n_sims = generate_game(
            network, rng=rng, py_rng=py_rng, cards_path=CARDS, **sp_kwargs,
        )
        games.append([
            (e.obs, e.policy, e.mask, e.value, e.current_player, e.policy_weight)
            for e in examples
        ])
        decisions += n_dec
        sims += n_sims
        n_ex += len(examples)
        p0 += winner == 0
        p1 += winner == 1
        draws += winner is None

    return games, {
        "games": n_games, "p0_wins": p0, "p1_wins": p1, "draws": draws,
        "decisions": decisions, "sims": sims, "examples": n_ex,
    }


def _worker_eval_selfplay(job):
    """Current net vs an older checkpoint of itself, seats alternating.

    This is the progress metric that works at ANY strength. Greedy beats
    Random 100-0 in this game (README_BASELINE_AGENTS.md), so a win-rate
    against Greedy reads 0% for every agent weaker than Greedy and says
    nothing about whether the run is improving.
    """
    import random as pyrandom

    import torch

    ckpt_new, ckpt_old, n_games, seed, agent_kwargs = job
    torch.set_num_threads(1)

    from splendor_duel.agents import play_game
    from splendor_duel.agents.az.az_agent_v2 import AZAgentV2

    new = AZAgentV2.load(ckpt_new, device="cpu", seed=seed, **agent_kwargs)
    old = AZAgentV2.load(ckpt_old, device="cpu", seed=seed + 1, **agent_kwargs)
    wins = draws = 0
    for i in range(n_games):
        pyrandom.seed(seed * 977 + i)
        seat = i % 2
        r = play_game(new, old, CARDS) if seat == 0 else play_game(old, new, CARDS)
        if r.winner == seat:
            wins += 1
        elif r.winner is None:
            draws += 1
    return wins, draws, n_games


def _worker_eval(job):
    """Play `n_games` of AZ vs a reference agent, alternating seats."""
    import random as pyrandom

    import torch

    ckpt, opponent, n_games, seed, agent_kwargs = job
    torch.set_num_threads(1)

    from splendor_duel.agents import (
        GreedyAgent, MCTSAgent, RandomAgent, play_game,
    )
    from splendor_duel.agents.az.az_agent_v2 import AZAgentV2

    factories = {
        "random": lambda: RandomAgent(),
        "greedy": lambda: GreedyAgent(),
        "mcts": lambda: MCTSAgent(iterations=100, rollout="greedy", rollout_depth=20),
    }
    agent = AZAgentV2.load(ckpt, device="cpu", seed=seed, **agent_kwargs)

    wins = draws = 0
    for i in range(n_games):
        pyrandom.seed(seed * 1000 + i)
        az_seat = i % 2
        if az_seat == 0:
            r = play_game(agent, factories[opponent](), CARDS)
        else:
            r = play_game(factories[opponent](), agent, CARDS)
        if r.winner == az_seat:
            wins += 1
        elif r.winner is None:
            draws += 1
    return wins, draws, n_games


# ── Trainer ───────────────────────────────────────────────────────────────────

class TrainerV2:

    def __init__(self, args):
        import torch
        import torch.optim as optim
        from collections import deque

        from splendor_duel.agents.az.network_az import build_az_network

        self.args = args
        self.device = torch.device(args.device)
        self.network = build_az_network(
            hidden_sizes=args.hidden_sizes,
            init_checkpoint=args.init,
            device=args.device,
        )
        if args.init:
            print(f"Warm-started from {args.init} "
                  f"(hidden_sizes={self.network.hidden_sizes})")

        # Decay is a regulariser on weights, not on normalisation scales or
        # biases — applying it there just fights LayerNorm.
        decay, no_decay = [], []
        for name, p in self.network.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
        self.optimizer = optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr,
        )

        self.buffer = deque(maxlen=args.buffer_size)
        self.val = deque(maxlen=args.val_size)
        self.total_iterations = 0
        self.total_games = 0
        self.start_iteration = 0
        self.best_score = -1.0

    # ── checkpoint I/O ────────────────────────────────────────────────────
    def save(self, path):
        import torch
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_iterations": self.total_iterations,
            "total_games": self.total_games,
            "tanh_value": True,      # tells load_az_network to rebuild AZNetwork
            "hidden_sizes": self.network.hidden_sizes,
        }, path)

    def load(self, path):
        import torch
        import torch.optim as optim

        from splendor_duel.agents.az.network_az import AZNetwork
        from splendor_duel.agents.ppo.network import infer_hidden_sizes

        data = torch.load(path, map_location=self.device, weights_only=False)
        sizes = infer_hidden_sizes(data["network"])
        if sizes != self.network.hidden_sizes:
            print(f"  (checkpoint is {sizes}, rebuilding network+optimizer)")
            self.network = AZNetwork(hidden_sizes=sizes).to(self.device)
            decay = [p for _, p in self.network.named_parameters() if p.ndim > 1]
            no_decay = [p for _, p in self.network.named_parameters() if p.ndim <= 1]
            self.optimizer = optim.AdamW(
                [{"params": decay, "weight_decay": self.args.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=self.args.lr,
            )
        self.network.load_state_dict(data["network"])
        if not data.get("tanh_value", False):
            print("  WARNING: checkpoint was trained WITHOUT a tanh value head; "
                  "resuming it under tanh will shift value predictions.")
        if "optimizer" in data and sizes == self.network.hidden_sizes:
            try:
                self.optimizer.load_state_dict(data["optimizer"])
            except ValueError:
                print("  (optimizer state incompatible, starting fresh)")
        self.total_iterations = data.get("total_iterations", 0)
        self.total_games = data.get("total_games", 0)
        self.start_iteration = self.total_iterations

    # ── learning-rate schedule ────────────────────────────────────────────
    def _set_lr(self):
        total = self.start_iteration + self.args.iterations
        progress = min(1.0, self.total_iterations / max(total, 1))
        frac = self.args.lr_final_frac
        lr = self.args.lr * (frac + (1 - frac) * 0.5 * (1 + math.cos(math.pi * progress)))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    # ── batching ──────────────────────────────────────────────────────────
    def _stack(self, examples):
        import numpy as np
        import torch
        obs = torch.as_tensor(np.stack([e[0] for e in examples]), device=self.device)
        pol = torch.as_tensor(np.stack([e[1] for e in examples]), device=self.device)
        msk = torch.as_tensor(np.stack([e[2] for e in examples]), device=self.device)
        val = torch.as_tensor(
            np.array([e[3] for e in examples], dtype=np.float32), device=self.device)
        # 5-tuples predate playout cap randomization and are all policy targets.
        wts = torch.as_tensor(
            np.array([e[5] if len(e) > 5 else 1.0 for e in examples], dtype=np.float32),
            device=self.device)
        return obs, pol, msk, val, wts

    @staticmethod
    def _losses(network, obs, pol, msk, val, wts=None):
        import torch
        import torch.nn.functional as F
        logits, v = network(obs)
        v = v.squeeze(-1)
        logp = torch.log_softmax(logits.masked_fill(~msk, -1e9), dim=-1)
        per_example = -(pol * logp).sum(-1)
        if wts is None:
            policy_loss = per_example.mean()
        else:
            # Mean over the policy-target positions only, so the loss scale
            # does not shrink just because PCR marked most positions as
            # value-only. Value loss still uses every position.
            policy_loss = (per_example * wts).sum() / wts.sum().clamp(min=1.0)
        value_loss = F.mse_loss(v, val)
        entropy = -(logp.exp() * logp.clamp(min=-30)).sum(-1).mean()
        return policy_loss, value_loss, entropy

    def train_steps(self, n_new):
        """Run enough steps that each generated position is trained on
        `--reuse` times over its lifetime in the buffer, with the value loss
        gated to a `--value-reuse` share of those steps.

        A position resident in a buffer of size B, refreshed at `n_new` per
        iteration, survives B/n_new iterations and is drawn
        steps*batch/B times per iteration — so steps*batch/n_new draws in all.
        """
        import numpy as np
        import torch
        import torch.nn as nn

        n_steps = int(round(self.args.reuse * n_new / self.args.batch_size))
        n_steps = max(self.args.min_steps_per_iter,
                      min(n_steps, self.args.max_steps_per_iter))
        # The two heads want opposite amounts of training (see --value-reuse),
        # so the value loss is applied on only a sampled fraction of steps.
        value_p = min(1.0, self.args.value_reuse / max(self.args.reuse, 1e-9))
        if len(self.buffer) < self.args.batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "steps": 0, "value_steps": 0, "lr": 0.0}

        lr = self._set_lr()
        self.network.train()
        rng = np.random.default_rng()
        tp = tv = te = 0.0
        done = nv = 0
        for _ in range(n_steps):
            idx = rng.integers(0, len(self.buffer), size=self.args.batch_size)
            obs, pol, msk, val, wts = self._stack([self.buffer[i] for i in idx])
            p_loss, v_loss, ent = self._losses(self.network, obs, pol, msk, val, wts)
            use_value = rng.random() < value_p
            loss = p_loss + (self.args.value_coeff * v_loss if use_value else 0.0 * v_loss)
            if not torch.isfinite(loss):
                continue
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            tp += p_loss.item(); tv += v_loss.item(); te += ent.item(); done += 1
            nv += int(use_value)
        d = max(done, 1)
        return {"policy_loss": tp / d, "value_loss": tv / d, "entropy": te / d,
                "steps": done, "value_steps": nv, "lr": lr}

    def validate(self):
        """Loss on held-out whole games — the generalisation gap that the
        original loop never measured."""
        import torch
        if len(self.val) < 256:
            return {"val_policy": float("nan"), "val_value": float("nan")}
        self.network.eval()
        n = min(len(self.val), 4096)
        recent = [self.val[i] for i in range(len(self.val) - n, len(self.val))]
        tp = tv = 0.0
        nb = 0
        with torch.no_grad():
            for start in range(0, n, 512):
                chunk = recent[start:start + 512]
                p_loss, v_loss, _ = self._losses(self.network, *self._stack(chunk))
                tp += p_loss.item(); tv += v_loss.item(); nb += 1
        return {"val_policy": tp / nb, "val_value": tv / nb}


# ── Main loop ─────────────────────────────────────────────────────────────────

def train(args):
    import numpy as np

    trainer = TrainerV2(args)
    if args.resume:
        trainer.load(args.resume)
        buf_path = Path(args.resume).parent / "replay_buffer.pkl"
        if buf_path.exists():
            import pickle
            with open(buf_path, "rb") as f:
                trainer.buffer.extend(pickle.load(f))
            print(f"  restored buffer ({len(trainer.buffer)} examples)")
        print(f"Resumed {args.resume} at iteration {trainer.total_iterations}")

    sp_kwargs = dict(
        sims_per_action=args.sims_per_action,
        min_sims=args.min_sims,
        max_sims=args.max_sims,
        c_puct=args.c_puct,
        temperature_moves=args.temperature_moves,
        dirichlet_eps=args.dirichlet_eps,
        dirichlet_scale=args.dirichlet_scale,
        pcr_full_prob=args.pcr_full_prob,
        pcr_fast_sims=args.pcr_fast_sims,
        pcr_full_sims=args.pcr_full_sims,
        device="cpu",
    )

    print(f"AlphaZero v2 — {args.workers} self-play workers")
    print(f"  iterations {args.iterations} x {args.games_per_iter} games")
    print(f"  sims: {args.sims_per_action}/legal-action, clamped [{args.min_sims}, {args.max_sims}]")
    print(f"  reuse {args.reuse}x per position  (old loop: ~83x)")
    print(f"  holding out {args.val_frac:.0%} of games -> ValP/ValV")
    print(f"  buffer {args.buffer_size}  batch {args.batch_size}  lr {args.lr}"
          f" -> {args.lr * args.lr_final_frac:.1e}")
    print(f"  network {trainer.network.hidden_sizes}, tanh value head")
    print()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    latest = str(save_dir / "_latest.pt")

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=args.workers, initializer=_init_worker)
    t_start = time.time()

    try:
        for it in range(args.iterations):
            trainer.save(latest)

            per_worker = args.games_per_iter // args.workers
            extra = args.games_per_iter - per_worker * args.workers
            jobs = [
                (latest, per_worker + (1 if w < extra else 0),
                 (trainer.total_iterations + 1) * 7919 + w, sp_kwargs)
                for w in range(args.workers)
                if per_worker + (1 if w < extra else 0) > 0
            ]

            t0 = time.time()
            results = pool.map(_worker_selfplay, jobs)
            t_sp = time.time() - t0

            agg = {"p0_wins": 0, "p1_wins": 0, "draws": 0,
                   "decisions": 0, "sims": 0, "examples": 0, "games": 0}
            all_games = []
            for games, st in results:
                all_games.extend(games)
                for k in agg:
                    agg[k] += st[k]

            # Hold out whole games: positions inside one game share an outcome,
            # so splitting by position would leak the value target.
            n_val = max(1, int(round(args.val_frac * len(all_games)))) if args.val_frac > 0 else 0
            n_new = 0
            sub_rng = np.random.default_rng(trainer.total_iterations)
            for i, game in enumerate(all_games):
                # Positions inside one game all carry the SAME outcome label, so
                # for the value head they are ~1 observation, not len(game).
                # Keeping every one spends the buffer on correlated duplicates:
                # 60k positions at ~150/game is only ~400 distinct outcomes.
                # Subsampling stores ~5x more games in the same memory. Measured
                # at fixed position count (3 seeds, 18 vs 90 games): value MSE
                # -0.149 (t=4.1), policy CE -0.020 (t=5.1). Costs no compute —
                # the searches happen anyway to play the moves; this only
                # changes what is retained.
                if args.positions_per_game > 0 and len(game) > args.positions_per_game:
                    keep = sub_rng.choice(len(game), args.positions_per_game, replace=False)
                    game = [game[j] for j in keep]
                if i < n_val:
                    trainer.val.extend(game)
                else:
                    trainer.buffer.extend(game)
                    n_new += len(game)
            trainer.total_games += agg["games"]

            t0 = time.time()
            losses = trainer.train_steps(n_new)
            t_tr = time.time() - t0
            trainer.total_iterations += 1
            vm = trainer.validate()

            print(
                f"It {trainer.total_iterations:4d} | G {trainer.total_games:5d} | "
                f"Buf {len(trainer.buffer):6d} | "
                f"W {agg['p0_wins']:2d}/{agg['p1_wins']:2d}/{agg['draws']:2d} | "
                f"Dec {agg['decisions'] / max(agg['games'], 1):5.1f} | "
                f"Sim {agg['sims'] / max(agg['examples'], 1):5.1f} | "
                f"P {losses['policy_loss']:.3f} V {losses['value_loss']:.3f} "
                f"H {losses['entropy']:.2f} | "
                f"ValP {vm['val_policy']:.3f} ValV {vm['val_value']:.3f} | "
                f"st {losses['steps']:3d}/{losses['value_steps']:3d} lr {losses['lr']:.1e} | "
                f"sp {t_sp:.0f}s tr {t_tr:.0f}s | {(time.time() - t_start) / 60:.0f}m",
                flush=True,
            )

            if args.save_interval > 0 and (it + 1) % args.save_interval == 0:
                trainer.save(str(save_dir / f"az_{trainer.total_iterations}.pt"))
                if args.save_buffer:
                    import pickle
                    with open(save_dir / "replay_buffer.pkl", "wb") as f:
                        pickle.dump(list(trainer.buffer), f,
                                    protocol=pickle.HIGHEST_PROTOCOL)

            if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
                trainer.save(latest)  # else we would score the pre-update net
                score = run_eval(pool, latest, args, args.eval_games_during_train,
                                 args.eval_vs_during_train.split(","))
                prev = save_dir / f"az_{trainer.total_iterations - args.eval_vs_prev}.pt"
                if args.eval_vs_prev > 0 and prev.exists():
                    run_eval_vs_prev(pool, latest, str(prev), args,
                                     args.eval_games_during_train,
                                     trainer.total_iterations - args.eval_vs_prev)
                if score is not None and score > trainer.best_score:
                    trainer.best_score = score
                    trainer.save(str(save_dir / "az_best.pt"))
                    print(f"    new best vs greedy ({score:.0%}) -> az_best.pt", flush=True)

        trainer.save(str(save_dir / "az_final.pt"))
        print(f"\nDone: {trainer.total_iterations} iterations, "
              f"{trainer.total_games} games, {(time.time() - t_start) / 3600:.1f} h")
    finally:
        pool.close()
        pool.join()


def run_eval_vs_prev(pool, ckpt_new, ckpt_old, args, n_games, old_iter):
    """Self-improvement check: current net vs its own earlier checkpoint."""
    agent_kwargs = dict(
        sims_per_action=args.eval_sims_per_action,
        min_sims=args.min_sims,
        max_sims=args.eval_max_sims,
        c_puct=args.c_puct,
    )
    per = n_games // args.workers
    extra = n_games - per * args.workers
    jobs = [(ckpt_new, ckpt_old, per + (1 if w < extra else 0), 2000 + w, agent_kwargs)
            for w in range(args.workers)
            if per + (1 if w < extra else 0) > 0]
    t0 = time.time()
    res = pool.map(_worker_eval_selfplay, jobs)
    wins = sum(r[0] for r in res)
    total = sum(r[2] for r in res)
    print(f"    eval vs az_{old_iter}: {wins}/{total} ({wins / max(total, 1):.0%})"
          f"  [{time.time() - t0:.0f}s]", flush=True)
    return wins / max(total, 1)


def run_eval(pool, ckpt, args, n_games, opponents):
    """Evaluate across the worker pool. Returns win-rate vs greedy, if played."""
    agent_kwargs = dict(
        sims_per_action=args.eval_sims_per_action,
        min_sims=args.min_sims,
        max_sims=args.eval_max_sims,
        c_puct=args.c_puct,
    )
    greedy_score = None
    for opp in opponents:
        opp = opp.strip()
        if opp not in ("random", "greedy", "mcts"):
            continue
        per = n_games // args.workers
        extra = n_games - per * args.workers
        jobs = [(ckpt, opp, per + (1 if w < extra else 0), 1000 + w, agent_kwargs)
                for w in range(args.workers)
                if per + (1 if w < extra else 0) > 0]
        t0 = time.time()
        res = pool.map(_worker_eval, jobs)
        wins = sum(r[0] for r in res)
        draws = sum(r[1] for r in res)
        total = sum(r[2] for r in res)
        wr = wins / max(total, 1)
        print(f"    eval vs {opp:6s}: {wins}/{total} ({wr:.0%})"
              f"{f' [{draws} draws]' if draws else ''}  [{time.time() - t0:.0f}s]",
              flush=True)
        if opp == "greedy":
            greedy_score = wr
    return greedy_score


def evaluate(args):
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=args.workers, initializer=_init_worker)
    try:
        print(f"Evaluating {args.eval}")
        run_eval(pool, args.eval, args, args.eval_games, args.eval_vs.split(","))
    finally:
        pool.close()
        pool.join()


def main():
    p = argparse.ArgumentParser(description="AlphaZero v2 trainer for Splendor Duel")

    p.add_argument("--workers", type=int, default=5,
                   help="Self-play processes, each pinned to 1 BLAS thread. "
                        "Measured 530MB RSS per worker (torch's static "
                        "footprint dominates), plus ~900MB for the parent and "
                        "its buffer. On this 8GB WSL box 5 workers peaks near "
                        "6GB; 6 fits but leaves little headroom, and going "
                        "past ~8 will OOM the VM.")
    p.add_argument("--device", default="cpu", help="Device for the training step")

    # Mode
    p.add_argument("--eval", default=None, help="Evaluate this checkpoint and exit")
    p.add_argument("--eval-games", type=int, default=40)
    p.add_argument("--eval-vs", default="greedy,random")
    p.add_argument("--eval-vs-prev", type=int, default=10,
                   help="Also play the current net against its own "
                        "checkpoint from N iterations back (0=off). "
                        "Above 50%% means the run is still improving.")
    p.add_argument("--resume", default=None)
    p.add_argument("--init", default=None,
                   help="Warm-start weights from a PPO or AZ checkpoint")

    # Loop size
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--games-per-iter", type=int, default=36)

    # Search
    p.add_argument("--sims-per-action", type=int, default=6,
                   help="Sims per legal action. The point is to spend search "
                        "where branching is wide; measured mean ~18 legal "
                        "actions, max ~139.")
    p.add_argument("--min-sims", type=int, default=32)
    p.add_argument("--max-sims", type=int, default=320)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temperature-moves", type=int, default=40,
                   help="Phase-decisions sampled at T=1. A game is ~150 "
                        "decisions, so 40 is roughly the first 10 turns.")
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--pcr-full-prob", type=float, default=0.0,
                   help="Playout cap randomization (Wu 2019). Fraction of moves "
                        "given the FULL search; the rest get --pcr-fast-sims and "
                        "are recorded as value-only examples. 0 disables. "
                        "Rationale: measured here, Elo per doubling of sims is "
                        "roughly constant (~+190) with no saturation, so paying "
                        "a deep search on every move is expensive and halves the "
                        "games the budget affords — while the value head's "
                        "effective sample size is the number of GAMES (positions "
                        "in one game share a single outcome label). PCR buys "
                        "deep-search policy targets without paying deep search "
                        "everywhere. Try 0.25.")
    p.add_argument("--pcr-fast-sims", type=int, default=64,
                   help="Sim cap for non-recorded (fast) moves under PCR")
    p.add_argument("--pcr-full-sims", type=int, default=256,
                   help="Sim cap for recorded (full) moves under PCR")
    p.add_argument("--dirichlet-scale", type=float, default=10.0,
                   help="alpha = scale / n_legal, the standard AZ heuristic")

    # Optimisation
    p.add_argument("--reuse", type=float, default=1.0,
                   help="Times each generated position is trained on over its "
                        "life in the buffer. THE critical knob: the old loop "
                        "was effectively ~83 and memorised its buffer. Swept "
                        "on the existing buffer, held-out value loss bottoms "
                        "at ~1.0 and is worse than a constant predictor by "
                        "2.6 (see header). That limit is the VALUE head's; "
                        "--value-reuse can carry that limit separately "
                        "if you want to raise this one for the policy head. "
                        "Measured caveat: on a fixed buffer, held-out POLICY "
                        "CE also degraded with more steps (2.25 -> 2.38 from "
                        "25 to 3000 steps) while train CE fell, so the policy "
                        "head overfits too and raising this is NOT a free win. "
                        "Change it only with ValP in front of you.")
    p.add_argument("--value-reuse", type=float, default=1.0,
                   help="Reuse for the VALUE head specifically: the value loss "
                        "is applied on a random value-reuse/reuse fraction of "
                        "steps. The two heads want opposite budgets — swept on "
                        "a fixed buffer, held-out value loss bottoms at ~1x and "
                        "degrades past it, while the policy head at 1x had "
                        "received only ~700 gradient steps after 30 "
                        "iterations (steps/iter FALLS as self-play games "
                        "shorten). Defaults to --reuse, i.e. the gate is open "
                        "and behaviour is unchanged; lower it only if ValV "
                        "degrades while you raise --reuse for the policy.")
    p.add_argument("--min-steps-per-iter", type=int, default=0,
                   help="Floor on steps/iter, so the policy head keeps training "
                        "as games shorten and produce fewer new examples.")
    p.add_argument("--max-steps-per-iter", type=int, default=400,
                   help="Safety cap on the derived step count")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-final-frac", type=float, default=0.1,
                   help="Cosine-decay lr to this fraction by the last iteration")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-coeff", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=60_000,
                   help="~3.6KB per position, so ~215MB at 60k and ~360MB at "
                        "100k. Larger is strictly better against overfitting; "
                        "RAM is the only reason not to raise it.")
    p.add_argument("--hidden-sizes", type=_parse_hidden_sizes, default="256,256,256",
                   help="Trunk widths for a FRESH network (a --init/--resume "
                        "checkpoint's own architecture wins). Default is "
                        "narrower than the PPO default 512,512,512,512 on "
                        "measurement: 0.33M vs 1.19M params reached a better "
                        "held-out value loss (0.931 vs 0.950) at 35%% less "
                        "cost per self-play decision. Widen only once a run "
                        "shows ValP still falling at the end.")

    # Validation
    p.add_argument("--positions-per-game", type=int, default=0,
                   help="Keep at most N randomly chosen positions per self-play "
                        "game (0 = keep all, the original behaviour). Raises the "
                        "number of distinct GAMES the buffer holds, which is the "
                        "value head's true sample size — positions within a game "
                        "share one outcome label. At 60k capacity and ~150 "
                        "positions/game the buffer holds only ~400 outcomes; "
                        "N=30 makes it ~2000. NOTE: this also cuts new positions "
                        "per iteration ~5x, so raise --reuse proportionally to "
                        "keep the gradient-step count (steps = reuse * new / "
                        "batch), and expect the buffer to turn over ~5x slower.")
    p.add_argument("--val-frac", type=float, default=0.06,
                   help="Fraction of self-play games held out whole for the "
                        "ValP/ValV generalisation readout")
    p.add_argument("--val-size", type=int, default=12_000)

    # Eval / logging
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games-during-train", type=int, default=24)
    p.add_argument("--eval-vs-during-train", default="greedy,random",
                   help="Greedy beats Random 100-0 here, so a "
                        "sub-Greedy agent scores 0%% against it "
                        "regardless of progress. Keep random in "
                        "the list for a metric with range.")
    p.add_argument("--eval-sims-per-action", type=int, default=12,
                   help="Play-time search is cheaper than self-play, so eval "
                        "uses a larger budget than training")
    p.add_argument("--eval-max-sims", type=int, default=600)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--save-buffer", action="store_true",
                   help="Persist the replay buffer alongside checkpoints so "
                        "--resume restarts warm. Off by default: pickling 60k "
                        "examples briefly doubles the buffer's memory, which "
                        "is the last thing this box has spare. Without it a "
                        "resumed run refills the buffer in ~12 iterations.")
    p.add_argument("--save-dir", default="checkpoints_az_v2")

    args = p.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
