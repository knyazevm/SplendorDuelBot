import random

random.seed(42)

from splendor_duel.agents import RandomAgent, GreedyAgent, MCTSAgent, play_game

result = play_game(
    MCTSAgent(iterations=200, rollout='none'),  # GreedyAgent(seed=1),
    MCTSAgent(iterations=100, rollout='greedy', rollout_depth=20),
    "data/cards.json",
    verbose=True,
)
print(result)
