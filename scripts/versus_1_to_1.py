import random

random.seed(42)

from splendor_duel.agents import RandomAgent, GreedyAgent, MCTSAgent, play_game

mcts = MCTSAgent(iterations=100, rollout='greedy', rollout_depth=20)

result = play_game(
    GreedyAgent(seed=1),
    mcts,
    "data/cards.json",
    verbose=True,
)
print(result)
