"""
gymnasium_env.py — Gymnasium environment for Splendor Duel.

Modes:
  - vs agent:   env plays one side, built-in opponent plays the other.
  - self-play:  env alternates, caller controls both sides.

Usage (vs agent):
    env = SplendorDuelEnv(opponent="greedy")
    obs, info = env.reset()
    while True:
        action = my_agent.pick(obs, info["legal_mask"])
        obs, reward, done, truncated, info = env.step(action)
        if done: break

Usage (self-play):
    env = SplendorDuelEnv(opponent="self")
    obs, info = env.reset()
    while True:
        action = current_policy.pick(obs, info["legal_mask"])
        obs, reward, done, truncated, info = env.step(action)
        if done: break
        # obs is now from the OTHER player's perspective
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState
from splendor_duel.game.actions import Phase
from splendor_duel.agents import BaseAgent, RandomAgent, GreedyAgent, MCTSAgent

from .action_map import (
    N_ACTIONS, action_to_index, index_to_action,
    legal_mask as compute_legal_mask,
    legal_actions_with_indices,
)
from .observation import OBS_SIZE, encode_state

CARDS_PATH = "data/cards.json"

# Hard ceiling on sub-actions inside ONE opponent turn.  A turn is bounded by
# the phase machine (<= 3 scrolls, refill, one main action, effect, royal,
# discards) and extra_turn chains are bounded by the finite decks, so a few
# dozen is already generous; this only catches a phase that fails to advance.
_OPPONENT_STEP_LIMIT = 1000


def outcome_reward(winner: Optional[int], player: int) -> float:
    """
    Terminal reward for `player`, given the recorded winner.

    Three-way on purpose.  `winner` is Optional[int], and every call site used
    to branch two ways — `1.0 if winner == me else -1.0` — which folded a
    winner-less game (None) into "loss" for BOTH players.  That breaks the
    antisymmetry a value function has to satisfy, v(s, p) == -v(s, 1 - p), and
    feeds the value head contradictory targets from one position.
    """
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0

_OPPONENT_FACTORIES = {
    "random": lambda: RandomAgent(),
    "greedy": lambda: GreedyAgent(),
    "mcts": lambda: MCTSAgent(iterations=100, rollout='none'),
}


class SplendorDuelEnv(gym.Env):
    """
    Gymnasium environment for Splendor Duel.

    Observation: float32 vector of length OBS_SIZE (519).
    Action: int in [0, N_ACTIONS) with masking via info["legal_mask"].
    Reward: +1.0 win, -1.0 loss, 0.0 otherwise.

    Parameters:
        opponent:    "random", "greedy", "mcts", or "self" (caller plays both)
        cards_path:  path to cards.json
        max_turns:   truncate game after this many turns (0 = no limit)
        reward_shaping: if True, add small intermediate rewards for progress
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
            self,
            opponent: str = "greedy",
            cards_path: str = CARDS_PATH,
            max_turns: int = 200,
            reward_shaping: bool = False,
            render_mode: str | None = None,
    ):
        super().__init__()
        self.cards_path = cards_path
        self.max_turns = max_turns
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Box(
            low=0.0, high=2.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        # Opponent setup
        self.self_play = (opponent == "self")
        self._opponent_factory = (
            None if self.self_play
            else _OPPONENT_FACTORIES.get(opponent)
        )
        if not self.self_play and self._opponent_factory is None:
            raise ValueError(
                f"Unknown opponent: {opponent}. "
                f"Available: {list(_OPPONENT_FACTORIES.keys()) + ['self']}"
            )

        # State (set in reset)
        self._state: Optional[GameState] = None
        self._opponent: Optional[BaseAgent] = None
        self._my_player: int = 0  # which player index the RL agent controls
        self._prev_points: float = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        import random as stdlib_random
        if seed is not None:
            stdlib_random.seed(seed)

        self._state = GameState.new_game(self.cards_path)

        if self.self_play:
            self._my_player = self._state.current_player
        else:
            # RL agent is always player 0 in terms of assignment,
            # but who goes first is random (set by new_game)
            self._my_player = 0
            self._opponent = self._opponent_factory()
            self._opponent.notify_game_start(1 - self._my_player)

            # If opponent goes first, run opponent turns
            if self._state.current_player != self._my_player:
                self._run_opponent()

            # Edge case: opponent won during opening → re-roll
            if self._state.is_game_over:
                return self.reset(seed=None, options=options)

        self._prev_points = float(self._state.players[self._my_player].points)

        obs = encode_state(self._state)
        info = self._make_info()

        # Safety: ensure mask is non-empty
        if info["legal_mask"].sum() == 0 and not self._state.is_game_over:
            return self.reset(seed=None, options=options)

        return obs, info

    def step(self, action: int):
        assert self._state is not None, "Call reset() first"
        assert not self._state.is_game_over, "Game is over, call reset()"

        mask = compute_legal_mask(self._state)
        if not mask[action]:
            # Illegal action — auto-pick a legal one with penalty
            legal_indices = np.where(mask)[0]
            if len(legal_indices) == 0:
                # No legal actions at all — treat as loss
                obs = encode_state(self._state)
                info = self._make_info()
                return obs, -1.0, True, False, info
            action = int(legal_indices[0])
            penalty = -10.0
        else:
            penalty = 0.0

        # Apply action (with defensive try/catch for edge cases)
        game_action = index_to_action(action)
        try:
            new_state = GameEngine.apply_action(self._state, game_action)
        except (IndexError, ValueError, KeyError):
            # Action was in mask but engine rejects it (rare race condition
            # or mask/engine mismatch). Fall back to a safe action.
            actions = GameEngine.get_legal_actions(self._state)
            if not actions:
                obs = encode_state(self._state)
                info = self._make_info()
                return obs, -1.0, True, False, info
            new_state = GameEngine.apply_action(self._state, actions[0])
            penalty -= 0.5
        self._state = new_state

        # Check if game ended
        if self._state.is_game_over:
            obs, reward, done, trunc, info = self._terminal_step()
            return obs, reward + penalty, done, trunc, info

        if self.self_play:
            obs = encode_state(self._state)
            reward = self._compute_reward() + penalty
            info = self._make_info()
            return obs, reward, False, self._truncated(), info

        # If it's still our turn (multi-phase: EFFECT, ROYAL, DISCARD),
        # return immediately for the next sub-action
        if self._state.current_player == self._my_player:
            obs = encode_state(self._state)
            reward = self._compute_reward() + penalty
            info = self._make_info()
            return obs, reward, False, self._truncated(), info

        # Opponent's turn — run until it's back to us (or game over)
        self._run_opponent()

        if self._state.is_game_over:
            obs, reward, done, trunc, info = self._terminal_step()
            return obs, reward + penalty, done, trunc, info

        obs = encode_state(self._state)
        reward = self._compute_reward() + penalty
        info = self._make_info()
        return obs, reward, False, self._truncated(), info

    def _truncated(self) -> bool:
        """
        Has the game run past max_turns?

        Applies on every non-terminal return path.  It used to be computed on
        one of them only — the branch taken after the built-in opponent moves —
        so `opponent="self"` never truncated at all and `max_turns` silently
        did nothing in that mode.  Games are not provably finite (tokens cycle
        board -> player -> bag -> board with nothing forcing progress), so the
        cap has to be real.
        """
        return self.max_turns > 0 and self._state.turn > self.max_turns

    def _run_opponent(self):
        """
        Run the opponent until control comes back to us, or the game ends.

        Returning while it is still the opponent's turn is the one thing this
        must never do: step() would then hand our policy an observation of the
        OPPONENT's position and let it play that side, generating transitions
        labelled with the wrong player.  Both former exits did exactly that,
        silently — an empty action list, and running out of the 200-iteration
        cap.  Neither is reachable now, so both are assertions instead.
        """
        for _ in range(_OPPONENT_STEP_LIMIT):
            if self._state.is_game_over:
                return
            if self._state.current_player == self._my_player:
                return
            actions = GameEngine.get_legal_actions(self._state)
            # The engine always offers something — PassTurn is the last resort
            # when nothing else is possible — so an empty list means the rules
            # engine is broken, not that the opponent is stuck.
            assert actions, (
                f"engine returned no legal actions in phase "
                f"{self._state.phase.name} for player "
                f"{self._state.current_player}"
            )
            action = self._opponent.choose_action(self._state, actions)
            self._state = GameEngine.apply_action(self._state, action)

        raise RuntimeError(
            f"opponent did not yield control within {_OPPONENT_STEP_LIMIT} "
            f"sub-actions (phase={self._state.phase.name}, "
            f"turn={self._state.turn}) — a single turn is bounded by the "
            f"phase machine and finite decks, so this means a phase loop"
        )

    def _terminal_step(self):
        """
        Final observation, reward, and the outcome for BOTH players.

        The returned scalar is the reward of the player whose move ended the
        game — the usual alternating-env convention.  In self-play that is
        structurally ALWAYS +1: the engine sets GAME_OVER without advancing
        `current_player`, so the mover is the winner by construction, and the
        losing side's transitions would otherwise never see a negative reward
        at all.  Callers that own both sides must therefore credit the loser
        from `info["final_rewards"]` rather than summing the scalar.
        """
        winner = self._state.winner
        final_rewards = {p: outcome_reward(winner, p) for p in (0, 1)}

        perspective = (
            self._state.current_player if self.self_play else self._my_player
        )
        reward = final_rewards[perspective]

        obs = encode_state(self._state)
        info = self._make_info()
        info["winner"] = winner
        info["final_rewards"] = final_rewards
        return obs, reward, True, False, info

    def _compute_reward(self) -> float:
        """Intermediate reward (0 if no shaping)."""
        if not self.reward_shaping:
            return 0.0
        me = self._state.players[self._my_player]
        curr_pts = float(me.points)
        delta = (curr_pts - self._prev_points) * 0.05  # small signal
        self._prev_points = curr_pts
        return delta

    def _make_info(self) -> dict:
        """Build info dict with legal mask and metadata."""
        mask = compute_legal_mask(self._state)
        return {
            "legal_mask": mask,
            "state": self._state,
            "current_player": self._state.current_player,
            "turn": self._state.turn,
            "phase": self._state.phase.name,
        }

    def render(self):
        if self.render_mode == "ansi":
            return repr(self._state)
        return None

    @property
    def state(self) -> GameState:
        return self._state
