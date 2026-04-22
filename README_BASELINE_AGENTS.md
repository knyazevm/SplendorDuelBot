# Фаза 2 — Baseline агенты: инструкция

## Игра с агентом
```
python scripts/play_web.py
```

Как играть:
- OPTIONAL фаза → кнопка "Proceed →" или клик по доске (если есть свитки)
- MAIN фаза → клик по 1-й ячейке доски, затем по 2-й → TakeTokens. Клик по карте пирамиды → popup "Buy" / "Reserve"
- EFFECT фаза → клик по доске (take_same_gem) или кнопки (take_opponent_gem, wildcard)
- ROYAL фаза → клик по royal-карте
- DISCARD фаза → клик по своему токену

## Структура файлов

Положить файлы в проект:

```
splendor_duel/agents/
├── __init__.py          # экспорты
├── base_agent.py        # абстрактный интерфейс
├── random_agent.py      # случайный агент
├── greedy_agent.py      # эвристический агент
├── mcts_agent.py        # чистый MCTS
└── game_runner.py       # play_game() + GameResult

scripts/
└── run_tournament.py    # турнир между агентами

tests/
└── test_agents.py       # 11 тестов
```

---

## Запуск тестов

```bash
# Все тесты агентов
pytest tests/test_agents.py -v

# Все тесты проекта
pytest tests/ -v

# С покрытием
pytest tests/ --cov=splendor_duel --cov-report=term-missing
```

Ожидаемый результат: **11 passed** (~30–40 секунд).

---

## Турнир: Random vs Greedy

```bash
# Быстрый (20 игр, ~6 секунд)
python scripts/run_tournament.py

# Статистически значимый (100 игр)
python scripts/run_tournament.py --games 100

# Другой seed
python scripts/run_tournament.py --games 50 --seed 123
```

Ожидаемый результат: Greedy побеждает Random ~95–100% игр.

---

## Турнир с MCTS

MCTS значительно медленнее (каждая игра ~30–120 секунд в зависимости от `--mcts-iters`).

```bash
# Быстрый: Greedy vs MCTS(none) — ~5 мин на 20 игр
python scripts/run_tournament.py --games 20 --include-mcts \
  --mcts-modes none --mcts-iters 200

# Сравнение всех режимов — ~30 мин
python scripts/run_tournament.py --games 20 --include-mcts \
  --mcts-modes none,greedy,random --mcts-iters 100

# Только MCTS(none) с разным бюджетом — запускай отдельно
python scripts/run_tournament.py --games 20 --include-mcts \
  --mcts-modes none --mcts-iters 100

python scripts/run_tournament.py --games 20 --include-mcts \
  --mcts-modes none --mcts-iters 400

# Чисто Greedy vs Random (быстро, ~10 сек)
python scripts/run_tournament.py --games 50
```

---

## Использование агентов в своём коде

### Одна игра

```python
import random
random.seed(42)

from splendor_duel.agents import RandomAgent, GreedyAgent, MCTSAgent, play_game

result = play_game(
    GreedyAgent(seed=1),
    RandomAgent(seed=2),
    "data/cards.json",
)
print(result)
# Game: Greedy vs Random → Player 0 (Greedy) wins by prestige_20 |
# scores=(20, 5) crowns=(4, 1) turns=48 (0.2s)
```

### Verbose-режим (лог каждого хода)

```python
result = play_game(
    GreedyAgent(),
    RandomAgent(),
    "data/cards.json",
    verbose=True,   # печатает каждое действие
)
```

### Свой агент

```python
from splendor_duel.agents import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="MyBot")

    def choose_action(self, state, legal_actions):
        # state — GameState (read-only!)
        # legal_actions — list[Action], всегда непустой
        # Вернуть один Action из списка
        return legal_actions[0]  # заглушка
```

### Настройка MCTS

```python
mcts = MCTSAgent(
    iterations=200,      # итераций дерева на ход (больше = сильнее, медленнее)
    exploration=1.41,    # UCB1 коэффициент исследования
    rollout_depth=40,    # глубина случайного доигрывания (шаги, не ходы)
    max_children=25,     # progressive widening (ограничение ветвления)
    time_limit=2.0,      # или лимит по времени в секундах (0 = без лимита)
    seed=42,             # для воспроизводимости
)
```

---

## GameResult — что возвращается

```python
@dataclass
class GameResult:
    winner: int             # 0 или 1
    victory_type: str       # 'prestige_20', 'crowns_10', 'mono_red', ...
    turns: int              # количество ходов (полных)
    steps: int              # количество шагов (включая под-фазы)
    scores: tuple[int, int] # финальные очки обоих игроков
    crowns: tuple[int, int] # финальные короны
    cards: tuple[int, int]  # количество купленных карт
    elapsed: float          # время игры в секундах
    agent_names: tuple[str, str]
```

---

## Результаты турнира (справочно)

| Matchup           | Win rate | Avg turns | Avg time |
|-------------------|----------|-----------|----------|
| Greedy vs Random  | 100%     | ~60       | ~0.3s    |

Greedy побеждает Random в 30/30 играх — эвристика работает. 
Основной путь к победе: `prestige_20` (набор 20 очков).

---

## Известные особенности

- **MCTS медленный** из-за высокого branching factor (~118 действий в MAIN-фазе на полной доске). Оптимизации: короткие rollouts (40 шагов) + heuristic eval + progressive widening (cap 25 детей). Для AlphaZero (Фаза 4) нейросеть заменит rollouts.
- **Greedy не использует reserve** активно — эвристика оценивает reserve с дисконтом 0.6x. Это сознательное упрощение.
- **Random агент детерминистичен** при указании seed — удобно для отладки.
