"""
Tests for Gymnasium environment: action_map, observation, gymnasium_env.
Run with:  pytest tests/test_env.py -v
"""
import json
import random

import numpy as np
import pytest

from splendor_duel.game.actions import (
    BuyCard, ChooseRoyal, DiscardToken, EffectChooseGold,
    EffectChooseWildcard, EffectSkip, PassTurn, ProceedToMain, RefillBoard,
    ReserveCard, TakeTokens, UseScroll, Phase,
)
from splendor_duel.game.constants import (
    ALL_SEGMENTS, EFFECT_CHOOSE_GOLD, GEM_NAMES, Gem, MAX_RESERVED, N_GEMS,
)
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
        assert N_ACTIONS == 278

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
        for action in [ProceedToMain(), RefillBoard(), EffectSkip(), PassTurn()]:
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
            ProceedToMain(), RefillBoard(), EffectSkip(), PassTurn(),
            TakeTokens(positions=ALL_SEGMENTS[0]),
            TakeTokens(positions=ALL_SEGMENTS[50]),
            BuyCard(source='pyramid', level=1, index=0),
            BuyCard(source='reserve', level=0, index=0),
            ReserveCard(source='pyramid', level=1, index=0),
            ReserveCard(source='deck', level=2, index=0),
            UseScroll(position=(0, 0)),
            ChooseRoyal(index=0),
            DiscardToken(gem=0),
            EffectChooseWildcard(colour=0),
        ]
        for a in actions:
            idx = action_to_index(a)
            assert idx not in seen, f"Collision at index {idx}"
            assert 0 <= idx < N_ACTIONS
            seen.add(idx)

    def test_wildcard_roundtrip(self):
        for colour in range(N_GEMS):
            action = EffectChooseWildcard(colour=colour)
            assert index_to_action(action_to_index(action)) == action

    def test_wildcard_stays_in_its_block(self):
        """Regression: wildcard indices used to spill into the ChooseRoyal block."""
        from splendor_duel.env.action_map import OFF_EFFECT_WC, OFF_ROYAL
        for colour in range(N_GEMS):
            idx = action_to_index(EffectChooseWildcard(colour=colour))
            assert OFF_EFFECT_WC <= idx < OFF_ROYAL

    def test_encoder_rejects_out_of_range_offsets(self):
        """_at() must raise rather than silently land in the next block."""
        for bad in [
            EffectChooseWildcard(colour=N_GEMS),
            ChooseRoyal(index=4),
            BuyCard(source='reserve', level=0, index=MAX_RESERVED),
        ]:
            with pytest.raises(AssertionError):
                action_to_index(bad)

    def test_roundtrip_over_real_games(self):
        """
        Every action the engine can actually emit must survive
        action_to_index → index_to_action unchanged, in every state of a real
        game.  This is the check that would have caught EffectChooseWildcard
        decoding as ChooseRoyal once a tableau passed 20 cards.
        """
        random.seed(20240607)
        for _ in range(60):
            state = GameState.new_game(CARDS_PATH)
            while not state.is_game_over:
                actions = GameEngine.get_legal_actions(state)
                if not actions:
                    break  # separate known bug: engine can emit zero actions
                seen: dict[int, object] = {}
                for a in actions:
                    idx = action_to_index(a)
                    assert 0 <= idx < N_ACTIONS
                    assert idx not in seen, f"{a} collides with {seen.get(idx)}"
                    seen[idx] = a
                    back = index_to_action(idx)
                    assert type(back) is type(a), (
                        f"{a} at index {idx} decoded as {back}"
                    )
                    # BuyCard-from-reserve carries an unused `level` the
                    # decoder cannot recover; the engine ignores it.
                    if not (isinstance(a, BuyCard) and a.source == 'reserve'):
                        assert back == a, f"{a} round-tripped to {back}"
                state = GameEngine.apply_action(state, random.choice(actions))

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


class TestOpponentHandback:
    """
    _run_opponent must never return while it is still the opponent's turn.

    If it does, step() hands our policy an observation of the opponent's
    position and lets it play that side, so the transition is recorded under
    the wrong player.  Silent, and it corrupts the value target.
    """

    def test_control_always_comes_back_to_us(self):
        for seed in range(20):
            env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
            _, info = env.reset(seed=seed)

            assert env.state.current_player == env._my_player, (
                f"reset() returned on the opponent's turn (seed {seed})"
            )

            done = False
            steps = 0
            while not done and steps < 400:
                mask = info["legal_mask"]
                action = int(np.random.choice(np.where(mask)[0]))
                _, _, done, truncated, info = env.step(action)
                steps += 1
                if done or truncated:
                    break
                assert env.state.current_player == env._my_player, (
                    f"step() returned on the opponent's turn "
                    f"(seed {seed}, step {steps})"
                )

    def test_empty_action_list_is_loud_not_silent(self):
        """A broken engine must raise, not quietly swap which side we play."""
        env = SplendorDuelEnv(opponent="random", cards_path=CARDS_PATH)
        env.reset(seed=3)
        # Force the opponent to be on turn, and make the engine look broken.
        env._my_player = 1 - env.state.current_player
        original = GameEngine.get_legal_actions
        try:
            GameEngine.get_legal_actions = staticmethod(lambda s: [])
            with pytest.raises(AssertionError, match="no legal actions"):
                env._run_opponent()
        finally:
            GameEngine.get_legal_actions = original


class TestTruncation:
    """max_turns has to apply in every mode, self-play included."""

    def test_self_play_truncates(self):
        env = SplendorDuelEnv(opponent="self", cards_path=CARDS_PATH,
                              max_turns=3)
        _, info = env.reset(seed=11)
        truncated = False
        for _ in range(400):
            action = int(np.where(info["legal_mask"])[0][0])
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert truncated, "self-play ignored max_turns"
        assert env.state.turn > 3

    def test_zero_disables_truncation(self):
        env = SplendorDuelEnv(opponent="self", cards_path=CARDS_PATH,
                              max_turns=0)
        _, info = env.reset(seed=11)
        for _ in range(200):
            action = int(np.where(info["legal_mask"])[0][0])
            _, _, done, truncated, info = env.step(action)
            assert not truncated
            if done:
                break


class TestObservationV2:
    def test_choose_wildcard_has_its_own_slot(self):
        """
        'choose_wildcard' is a pending_effect the engine invents; it is not a
        card ability, so it is absent from _ABILITY_IDS.  A .get(..., 0)
        lookup used to fold it onto slot 0 — the same encoding as "nothing
        pending" — leaving the network to choose a bonus colour from a
        position that looked like it had no decision to make.
        """
        from splendor_duel.env.observation_v2 import (
            OBS_SIZE_V2, _N_PENDING_EFFECT, _PENDING_EFFECT_IDS,
            encode_state_v2,
        )
        from splendor_duel.env.observation import OBS_SIZE

        assert _PENDING_EFFECT_IDS['choose_wildcard'] != _PENDING_EFFECT_IDS[None]
        assert len(set(_PENDING_EFFECT_IDS.values())) == _N_PENDING_EFFECT

        base = OBS_SIZE + 3 * 17  # opponent reserved block precedes it
        state = GameState.new_game(CARDS_PATH)

        none_block = encode_state_v2(state)[base:base + _N_PENDING_EFFECT]

        state.pending_effect = 'choose_wildcard'
        wc_block = encode_state_v2(state)[base:base + _N_PENDING_EFFECT]

        assert not np.array_equal(none_block, wc_block)
        assert wc_block.sum() == 1.0
        assert encode_state_v2(state).shape == (OBS_SIZE_V2,)

    def test_unknown_pending_effect_raises(self):
        from splendor_duel.env.observation_v2 import encode_state_v2

        state = GameState.new_game(CARDS_PATH)
        state.pending_effect = 'some_future_ability'
        with pytest.raises(KeyError, match="_PENDING_EFFECT_IDS"):
            encode_state_v2(state)

    def test_every_engine_pending_effect_is_encodable(self):
        """Play real games and assert the table covers what the engine sets."""
        from splendor_duel.env.observation_v2 import encode_state_v2

        random.seed(2024)
        seen = set()
        for _ in range(40):
            state = GameState.new_game(CARDS_PATH)
            for _ in range(600):
                if state.is_game_over:
                    break
                seen.add(state.pending_effect)
                encode_state_v2(state)  # raises if the effect is unknown
                actions = GameEngine.get_legal_actions(state)
                state = GameEngine.apply_action(state, random.choice(actions))
        assert 'choose_wildcard' in seen, "never exercised the wildcard path"


class TestChooseGold:
    """
    Reserving takes a gold from the board, and WHICH gold is a real choice:
    the tokens are interchangeable, the cells they vacate are not.  The engine
    used to take gold_positions[0] unconditionally.
    """

    def _reserve_state(self, seed):
        """Advance a game to a state where reserving is legal."""
        random.seed(seed)
        state = GameState.new_game(CARDS_PATH)
        for _ in range(400):
            actions = GameEngine.get_legal_actions(state)
            reserves = [a for a in actions if isinstance(a, ReserveCard)]
            golds = [(r, c) for r in range(5) for c in range(5)
                     if state.board.token_at(r, c) == Gem.GOLD]
            if reserves and len(golds) > 1:
                return state, reserves[0], golds
            if state.is_game_over:
                break
            state = GameEngine.apply_action(state, random.choice(actions))
        return None, None, None

    def test_reserve_asks_which_gold(self):
        state, reserve, golds = self._reserve_state(7)
        assert state is not None, "never reached a multi-gold reserve"

        after = GameEngine.apply_action(state, reserve)
        assert after.phase == Phase.EFFECT
        assert after.pending_effect == EFFECT_CHOOSE_GOLD

        offered = GameEngine.get_legal_actions(after)
        assert {a.position for a in offered} == set(golds)
        assert all(isinstance(a, EffectChooseGold) for a in offered)
        # The card is already reserved; only the gold is outstanding.
        assert len(after.active.reserved) == len(state.active.reserved) + 1
        assert after.active.tokens[Gem.GOLD] == state.active.tokens[Gem.GOLD]

    def test_choice_changes_the_board(self):
        state, reserve, golds = self._reserve_state(7)
        after = GameEngine.apply_action(state, reserve)

        boards = []
        for pos in golds:
            done = GameEngine.apply_action(after, EffectChooseGold(position=pos))
            assert done.pending_effect is None
            # token_at returns None for an empty cell, not the EMPTY sentinel.
            assert done.board.token_at(*pos) is None
            boards.append(done.board.grid.tobytes())
        assert len(set(boards)) == len(golds), "gold choice left the board identical"

    def test_single_gold_is_not_asked(self):
        """One gold on the board is not a decision — resolve it inline."""
        random.seed(21)
        checked = 0
        for _ in range(40):
            state = GameState.new_game(CARDS_PATH)
            for _ in range(400):
                if state.is_game_over:
                    break
                actions = GameEngine.get_legal_actions(state)
                reserves = [a for a in actions if isinstance(a, ReserveCard)]
                golds = [(r, c) for r in range(5) for c in range(5)
                         if state.board.token_at(r, c) == Gem.GOLD]
                if reserves and len(golds) == 1:
                    mover = state.current_player
                    after = GameEngine.apply_action(state, reserves[0])
                    assert after.pending_effect != EFFECT_CHOOSE_GOLD
                    # The turn has ended, so `after.active` is the OPPONENT —
                    # index the mover explicitly.
                    assert after.players[mover].tokens[Gem.GOLD] == \
                        state.players[mover].tokens[Gem.GOLD] + 1
                    checked += 1
                    break
                state = GameEngine.apply_action(state, random.choice(actions))
            if checked >= 5:
                break
        assert checked >= 5, "never observed a single-gold reserve"

    def test_gold_conservation_over_real_games(self):
        """Splitting reserve into two steps must not create or lose gold."""
        random.seed(31)
        for _ in range(30):
            state = GameState.new_game(CARDS_PATH)
            for _ in range(600):
                if state.is_game_over:
                    break
                total = (int(state.board.grid[state.board.grid == Gem.GOLD].size)
                         + int(state.players[0].tokens[Gem.GOLD])
                         + int(state.players[1].tokens[Gem.GOLD])
                         + state.bag['gold'])
                assert total == 3, f"gold count is {total}, not 3"
                actions = GameEngine.get_legal_actions(state)
                state = GameEngine.apply_action(state, random.choice(actions))


class TestWinner:
    def test_winner_is_recorded_not_recomputed(self):
        random.seed(5)
        checked = 0
        for _ in range(30):
            state = GameState.new_game(CARDS_PATH)
            for _ in range(800):
                if state.is_game_over:
                    break
                state = GameEngine.apply_action(
                    state, random.choice(GameEngine.get_legal_actions(state)))
            if state.is_game_over:
                # The engine ends the game without advancing current_player,
                # so the winner is whoever just moved.
                assert state.winner == state.current_player
                assert state.players[state.winner].check_victory() is not None
                checked += 1
        assert checked > 0

    def test_winner_survives_copy(self):
        state = GameState.new_game(CARDS_PATH)
        assert state.winner is None
        state.winner = 1
        assert state.copy().winner == 1

    def test_outcome_reward_is_three_way(self):
        from splendor_duel.env.gymnasium_env import outcome_reward

        assert outcome_reward(0, 0) == 1.0
        assert outcome_reward(0, 1) == -1.0
        assert outcome_reward(1, 1) == 1.0
        # A draw is 0 for BOTH, not a loss for both.
        assert outcome_reward(None, 0) == 0.0
        assert outcome_reward(None, 1) == 0.0
        # Antisymmetry: v(s, p) == -v(s, 1-p) for every decided outcome.
        for w in (0, 1):
            assert outcome_reward(w, 0) == -outcome_reward(w, 1)

    def test_self_play_terminal_reports_both_sides(self):
        """
        step() reports the mover's reward, which in self-play is always +1
        because the game ends on the winner's move.  The loser's outcome has
        to be reachable, or the terminal signal is a constant.
        """
        env = SplendorDuelEnv(opponent="self", cards_path=CARDS_PATH,
                              max_turns=0)
        _, info = env.reset(seed=13)
        for _ in range(800):
            action = int(np.random.choice(np.where(info["legal_mask"])[0]))
            _, reward, done, _, info = env.step(action)
            if done:
                finals = info["final_rewards"]
                assert set(finals) == {0, 1}
                assert sorted(finals.values()) == [-1.0, 1.0]
                assert finals[info["winner"]] == 1.0
                assert reward == 1.0  # the documented convention
                return
        pytest.fail("game did not finish")

    def test_buffer_credits_both_players(self):
        from splendor_duel.agents.ppo.trainer import RolloutBuffer

        buf = RolloutBuffer()
        # Two episodes in one rollout; only the second must be rewritten.
        for actor, done in [(0, 0), (1, 0), (0, 1),      # episode 1
                            (1, 0), (0, 0), (1, 0), (0, 1)]:  # episode 2
            buf.add(obs=None, action=0, log_prob=0.0, value=0.0,
                    reward=0.0, done=done, mask=None, actor=actor)
        buf.assign_final_rewards({0: 1.0, 1: -1.0})

        assert buf.rewards[6] == 1.0   # player 0's last transition
        assert buf.rewards[5] == -1.0  # player 1's last transition
        assert buf.rewards[:5] == [0.0] * 5, "reached into the previous episode"
