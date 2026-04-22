# SplendorDuelBot

Reinforcement learning agent for **Splendor Duel** (2-player board game).  
Goal: train an agent capable of beating a human player, and quantify the luck vs. skill ratio.

---

## Project structure

```
SplendorDuelBot/
├── data/
│   └── cards.json                  # full card database (all 67 cards + 4 royals)
├── images/
│   ├── cards/                      # card images — L1_01.jpg … R_04.jpg
│   ├── gems/                       # gem_white.png … gem_gold.png
│   └── tokens/
│       └── scroll.png
├── splendor_duel/
│   ├── game/
│   │   ├── constants.py            # Gem enum, board spiral, game limits
│   │   ├── card.py                 # Card / RoyalCard dataclasses + JSON loader
│   │   ├── board.py                # 5×5 numpy board, legal lines, refill
│   │   ├── actions.py              # action dataclasses (TakeTokens, Buy, Reserve…)
│   │   ├── player.py               # PlayerState
│   │   ├── state.py                # GameState + copy()
│   │   └── engine.py               # get_legal_actions() / apply_action()
│   ├── env/
│   │   ├── gymnasium_env.py        # Gymnasium wrapper
│   │   └── observation.py          # state → tensor encoding
│   └── agents/
│       ├── random_agent.py
│       ├── greedy_agent.py
│       └── alphazero/              # MCTS + policy/value network
├── training/
│   └── train.py
├── analysis/
│   └── luck_vs_skill.py
├── tests/
│   ├── test_board.py
│   ├── test_engine.py
│   └── test_env.py
├── requirements.txt
└── README.md
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1.1 Card database | ✅ | `data/cards.json` — all cards annotated |
| 1.2 Game engine | 🔄 | Pure-Python logic, fully tested |
| 1.3 Gymnasium env | ⬜ | Observation + action space |
| 2 Baseline agents | ⬜ | Random, Greedy, pure MCTS |
| 3 AlphaZero | ⬜ | Self-play + policy/value net |
| 4 Analysis | ⬜ | Luck vs skill experiments |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                # install package in editable mode
```

---

## Running tests

```bash
export PYTHONPATH=.
pytest 
```

```bash
# Single module
pytest tests/test_board.py -v

# With coverage
pytest tests/ --cov=splendor_duel --cov-report=term-missing
```

---

## Card database format

`data/cards.json` — two top-level keys: `cards` and `royal_cards`.

```json
{
  "cards": [
    {
      "id": "L1_01",
      "level": 1,
      "cost":      {"white": 2, "black": 0, "red": 1, "blue": 0, "green": 0, "pearl": 0, "gold": 0},
      "gem_bonus": {"green": 1},
      "points": 0,
      "crowns": 1,
      "ability": null
    }
  ],
  "royal_cards": [
    {
      "id": "R_01",
      "type": "royal",
      "points": 2,
      "ability": "take_scroll"
    }
  ]
}
```

**`gem_bonus`** values:
- `{"color": 1}` — standard single bonus
- `{"color": 2}` — double bonus
- `{"wildcard": 1}` — copies bonus of a card already in your tableau
- `null` — no bonus

**`ability`** values: `null` | `"extra_turn"` | `"take_same_gem"` | `"take_scroll"` | `"take_opponent_gem"`

---

## Victory conditions

| # | Condition |
|---|---|
| 1 | ≥ 20 prestige points (across all card colours) |
| 2 | ≥ 10 crowns |
| 3 | ≥ 10 prestige points on cards sharing one bonus colour |

---

## Key design decisions

- **Immutable state** — every `apply_action()` returns a new `GameState`; safe for MCTS tree search.
- **Numpy board** — 5×5 `int8` array; legal line generation uses vectorised indexing over pre-computed segments.
- **Observation modes** — full-information and partial-information (hidden reserves) both supported; set via env config.
- **Phase-based turns** — turn split into `OPTIONAL → MAIN → EFFECT → DISCARD` phases; each has its own `get_legal_actions()`.

## Использование среды Gymnasium
```
from splendor_duel.env import SplendorDuelEnv

# Против Greedy
env = SplendorDuelEnv(opponent="greedy")
obs, info = env.reset(seed=42)
mask = info["legal_mask"]  # bool[265]

# Выбрать действие (только где mask==True)
action = np.random.choice(np.where(mask)[0])
obs, reward, done, truncated, info = env.step(action)

# Self-play (для AlphaZero)
env = SplendorDuelEnv(opponent="self")
```

## Обучение PPO

```
# 1. Быстрый тест (~5 мин): должен начать побеждать Random
python scripts/train_ppo.py --opponent random --steps 50000 --device cuda

# 2. Основное обучение vs Greedy (~1-2 часа на CPU)
python scripts/train_ppo.py --opponent greedy --steps 500000

# 3. С GPU (если есть)
python scripts/train_ppo.py --opponent greedy --steps 500000 --device cuda

# 4. Оценка обученной модели
python scripts/train_ppo.py --eval checkpoints/ppo_final.pt --eval-games 50

# 5. Дообучение с чекпоинта
python scripts/train_ppo.py --resume checkpoints/ppo_250.pt --steps 1000000

# 6. Играть в web-интерфейсе — добавь PPOAgent в server.py AGENTS
```