import random

random.seed(42)

from splendor_duel.agents import GreedyAgent, MCTSAgent, play_game
from splendor_duel.agents.ppo import PPOAgent

result = play_game(
    PPOAgent.load("checkpoints/ppo_200.pt"),
    MCTSAgent(iterations=100, rollout='greedy', rollout_depth=20),
    "data/cards.json",
    verbose=True,
)
print(result)
