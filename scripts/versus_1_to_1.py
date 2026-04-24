import random

random.seed(42)

from splendor_duel.agents import GreedyAgent, GreedyByChatGPT, GeminiGreedyAgent, MCTSAgent, play_game
from splendor_duel.agents.ppo import PPOAgent
from splendor_duel.agents import (
    RandomAgent, GreedyAgent,
    GreedyByChatGPT, GeminiGreedyAgent, GreedyByClaude,
    GreedyByChatGPTV2, GeminiGreedyAgentV2, GreedyByClaudeV2,
    MCTSAgent, play_game, GameResult,
)

main_character = MCTSAgent(iterations=100, rollout='greedy', rollout_depth=20)

others = [
    GreedyAgent(),
    GreedyByChatGPT(seed=1),
    GeminiGreedyAgent(seed=1),
    GreedyByClaude(seed=1),
    GreedyByChatGPTV2(seed=1),
    GeminiGreedyAgentV2(seed=1),
    GreedyByClaudeV2(seed=1),
]

print("Ladder:")
print("\n".join(["- " + other_i.name for other_i in others]))

# Games
for i, other_i in enumerate(others):
    print(f"\nFight {i+1} / {len(others)}")
    result = play_game(
        main_character,
        other_i,
        "data/cards.json",
        verbose=False,
    )
    print(result)
