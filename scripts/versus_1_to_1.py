import random

random.seed(42)

from splendor_duel.agents import GreedyAgent, GreedyByChatGPT, GeminiGreedyAgent, MCTSAgent, play_game
from splendor_duel.agents.ppo import PPOAgent

games = 20

# Game 1
for _ in range(games):
    result = play_game(
        GreedyAgent(),
        GreedyByChatGPT(seed=1),
        "data/cards.json",
        verbose=False,
    )
    print(result)

# Game 2
for _ in range(games):
    result = play_game(
        GreedyAgent(),
        GeminiGreedyAgent(seed=1),
        "data/cards.json",
        verbose=False,
    )

    print(result)
