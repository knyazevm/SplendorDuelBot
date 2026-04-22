"""
Tests for Gymnasium environment: action_map, observation, gymnasium_env.
Run with:  pytest tests/test_env.py -v
"""
import json
import random

import numpy as np
import pytest

from splendor_duel.game.actions import (
    BuyCard, ChooseRoyal, DiscardToken, EffectSkip,
    ProceedToMain, RefillBoard, ReserveCard, TakeTokens,
    UseScroll, Phase,
)
from splendor_duel.game.constants import ALL_SEGMENTS, GEM_NAMES, Gem, N_GEMS
from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState

from splendor_duel.env import (
    SplendorDuelEnv, OBS_SIZE, N_ACTIONS,
    encode_state, action_to_index, index_to_action, legal_mask,
)

CARDS_PATH = "data/cards.json"


# ══════════════════════════════════════════════════════════════════════════════
# action_map
# ══════════════════════════════════════════════════════════════════════════════

class TestActionMap:

    def test_n_actions(self):
        assert N_ACTIONS == 265

    def test_take_tokens_roundtrip(self):
        for seg in ALL_SEGMENTS[:10]:
            action = TakeTokens(positions=seg)
            idx = action_to_index(action)
            back = index_to_action(idx)
            assert isinstance(back, TakeTokens)
            assert back.positions == seg

    def test_buy_pyramid_roundtrip(self):
        action = BuyCard(source='pyramid', level=2, index=1)
        idx = action_to_index(action)
        back = index_to_action(idx)
        assert isinstance(back, BuyCard)
        assert back.source == 'pyramid'
        assert back.level == 2 and back.index == 1

    def test_buy_reserve_roundtrip(self):
        action = BuyCard(source='reserve', level=0, index=2)
        idx = action_to_index(action)
        back = index_to_action(idx)
        assert isinstance(back, BuyCard)
        assert back.source == 'reserve'
        assert back.index == 2

    def test_reserve_pyramid_roundtrip(self):
        action = ReserveCard(source='pyramid', level=3, index=0)
        idx = action_to_index(action)
        back = index_to_action(idx)
        assert isinstance(back, ReserveCard)
        assert back.level == 3

    def test_reserve_deck_roundtrip(self):
        for lvl in (1, 2, 3):
            action = ReserveCard(source='deck', level=lvl, index=0)
            idx = action_to_index(action)
            back = index_to_action(idx)
            assert isinstance(back, ReserveCard)
            assert back.source == 'deck'
            assert back.level == lvl

    def test_scroll_roundtrip(self):
        action = UseScroll(position=(3, 4))
        idx = action_to_index(action)
        back = index_to_action(idx)
        assert isinstance(back, UseScroll)
        assert back.position == (3, 4)

    def test_special_actions_roundtrip(self):
        for action in [ProceedToMain(), RefillBoard(), EffectSkip()]:
            idx = action_to_index(action)
            back = index_to_action(idx)
            assert type(back) == type(action)

    def test_discard_roundtrip(self):
        for gem in range(N_GEMS):
            action = DiscardToken(gem=gem)
            idx = action_to_index(action)
            back = index_to_action(idx)
            assert isinstance(back, DiscardToken)
            assert back.gem == gem

    def test_no_index_collisions(self):
        """All action types map to distinct index ranges."""
        seen = set()
        # Sample actions from each type
        actions = [
            ProceedToMain(), RefillBoard(), EffectSkip(),
            TakeTokens(positions=ALL_SEGMENTS[0]),
            TakeTokens(positions=ALL_SEGMENTS[50]),
            BuyCard(source='pyramid', level=1, index=0),
            BuyCard(source='reserve', level=0, index=0),
            ReserveCard(source='pyramid', level=1, index=0),
            ReserveCard(source='deck', level=2, index=0),
            UseScroll(position=(0, 0)),
            ChooseRoyal(index=0),
            DiscardToken(gem=0),
        ]
        for a in actions:
            idx = action_to_index(a)
            assert idx not in seen, f"Collision at index {idx}"
            assert 0 <= idx < N_ACTIONS
            seen.add(idx)

    def test_legal_mask_shape(self):
        random.seed(42)
        state = GameState.new_game(CARDS_PATH)
        mask = legal_mask(state)
        assert mask.shape == (N_ACTIONS,)
        assert mask.dtype == np.bool_
        assert mask.sum() > 0

    def test_legal_mask_matches_engine(self):
        """Every legal action should have mask=True, and vice versa."""
        random.seed(42)
        state = GameState.new_game(CARDS_PATH)
        for _ in range(100):
            if state.is_game_over:
                break
            actions = GameEngine.get_legal_actions(state)
            mask = legal_mask(state)
            # Check all legal actions are in mask
            for a in actions:
                idx = action_to_index(a)
                assert mask[idx], f"Legal action {a} not in mask at index {idx}"
            # Check mask count matches
            assert mask.sum() == len(actions), (
                f"Mask has {mask.sum()} true, but {len(actions)} legal actions"
            )
            # Take random action and advance
            action = random.choice(actions)
            state = GameEngine.apply_action(state, action)


# ══════════════════════════════════════════════════════════════════════════════
# observation
# ══════════════════════════════════════════════════════════════════════════════

class TestObservation:

    def test_obs_size(self):
        assert OBS_SIZE == 519

    def test_encode_shape(self):
        random.seed(42)
        state = GameState.new_game(CARDS_PATH)
        obs = encode_state(state)
        assert obs.shape == (OBS_SIZE,)
        assert obs.dtype == np.float32

    def test_encode_values_bounded(self):
        """Observation values should be roughly in [0, 2]."""
        random.seed(42)
        state = GameState.new_game(CARDS_PATH)
        obs = encode_state(state)
        assert obs.min() >= -0.01, f"Min value: {obs.min()}"
        assert obs.max() <= 2.01, f"Max value: {obs.max()}"

    def test_encode_deterministic(self):
        random.seed(42)
        s1 = GameState.new_game(CARDS_PATH)
        obs1 = encode_state(s1)
        obs2 = encode_state(s1)
        np.testing.assert_array_equal(obs1, obs2)

    def test_encode_changes_after_action(self):
        random.seed(42)
        state = GameState.new_game(CARDS_PATH)
        obs_before = encode_state(state)
        actions = GameEngine.get_legal_actions(state)
        state2 = GameEngine.apply_action(state, actions[0])
        obs_after = encode_state(state2)
        assert not np.array_equal(obs_before, obs_after)

    def test_encode_through_full_game(self):
        """Encode every state in a random game — should never crash."""
        random.seed(42)
        state = GameState.new_game(CARDS_PATH)
        for _ in range(500):
            if state.is_game_over:
                break
            obs = encode_state(state)
            assert obs.shape == (OBS_SIZE,)
            assert np.all(np.isfinite(obs))
            actions = GameEngine.get_legal_actions(state)
            state = GameEngine.apply_action(state, random.choice(actions))


# ══════════════════════════════════════════════════════════════════════════════
# gymnasium_env
# ══════════════════════════════════════════════════════════════════════════════

class TestGymnasiumEnv:

    def test_reset_returns_valid_obs(self):
        env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBS_SIZE,)
        assert "legal_mask" in info
        assert info["legal_mask"].shape == (N_ACTIONS,)
        assert info["legal_mask"].sum() > 0

    def test_step_with_legal_action(self):
        env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
        obs, info = env.reset(seed=42)
        mask = info["legal_mask"]
        action = int(np.where(mask)[0][0])  # first legal action
        obs2, reward, done, truncated, info2 = env.step(action)
        assert obs2.shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_illegal_action_penalty(self):
        env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
        obs, info = env.reset(seed=42)
        mask = info["legal_mask"]
        # Find an illegal action
        illegal = int(np.where(~mask)[0][0])
        obs2, reward, done, truncated, info2 = env.step(illegal)
        assert reward == -10.0
        assert not done  # game continues

    def test_full_game_terminates(self):
        env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
        obs, info = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 2000:
            mask = info["legal_mask"]
            action = int(np.where(mask)[0][0])
            obs, reward, done, truncated, info = env.step(action)
            if truncated:
                break
            steps += 1
        assert done or truncated, f"Game didn't end in {steps} steps"

    def test_winner_reward(self):
        """Play a full game and check terminal reward is +1 or -1."""
        env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
        obs, info = env.reset(seed=42)
        done = False
        last_reward = 0.0
        while not done:
            mask = info["legal_mask"]
            action = int(np.where(mask)[0][0])
            obs, reward, done, truncated, info = env.step(action)
            last_reward = reward
            if truncated:
                break
        if done:
            assert last_reward in (1.0, -1.0), f"Terminal reward: {last_reward}"

    def test_greedy_opponent(self):
        env = SplendorDuelEnv(opponent="greedy", cards_path=CARDS_PATH)
        obs, info = env.reset(seed=42)
        mask = info["legal_mask"]
        assert mask.sum() > 0  # we have legal actions to play

    def test_self_play_mode(self):
        env = SplendorDuelEnv(opponent="self", cards_path=CARDS_PATH)
        obs, info = env.reset(seed=42)
        # In self-play, we control both sides
        done = False
        steps = 0
        players_seen = set()
        while not done and steps < 500:
            mask = info["legal_mask"]
            players_seen.add(info["current_player"])
            action = int(np.where(mask)[0][0])
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            if truncated:
                break
        # Should have seen both players
        assert len(players_seen) == 2

    def test_multiple_games(self):
        """Play 5 full games — all must work without errors."""
        env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            done = False
            while not done:
                mask = info["legal_mask"]
                # Pick random legal action
                legal_indices = np.where(mask)[0]
                action = int(np.random.choice(legal_indices))
                obs, reward, done, truncated, info = env.step(action)
                if truncated:
                    break

    def test_observation_space_contains(self):
        env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_action_space(self):
        env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
        assert env.action_space.n == N_ACTIONS
