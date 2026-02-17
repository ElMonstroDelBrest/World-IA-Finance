"""Tests for Strate IV: AsymmetricReward, LatentCryptoEnv."""

import numpy as np
import pytest
import torch

from src.strate_iv.config import EnvConfig, StrateIVConfig, load_config
from src.strate_iv.reward import AsymmetricReward
from src.strate_iv.trajectory_buffer import TrajectoryBuffer, TrajectoryEntry
from src.strate_iv.env import LatentCryptoEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dummy_entry(
    n_futures: int = 4,
    n_tgt: int = 8,
    patch_len: int = 16,
    d_model: int = 128,
    seq_len: int = 64,
    ctx_len: int = 256,
) -> TrajectoryEntry:
    """Create a synthetic TrajectoryEntry for testing."""
    # Simulate OHLCV with positive prices around 100
    future_ohlcv = torch.rand(n_futures, n_tgt, patch_len, 5) * 10 + 95

    return TrajectoryEntry(
        context_tokens=torch.randint(0, 1024, (seq_len,)),
        weekend_mask=torch.zeros(seq_len),
        context_ohlcv=torch.rand(ctx_len, 5) * 100 + 50,
        future_ohlcv=future_ohlcv,
        future_latents=torch.randn(n_futures, n_tgt, d_model),
        revin_means=torch.zeros(1, 5),
        revin_stds=torch.ones(1, 5) * 0.02,  # 2% vol
        last_close=100.0,
        h_x_pooled=torch.randn(d_model),
    )


def make_buffer(n_entries: int = 5, **kwargs) -> TrajectoryBuffer:
    """Create a buffer with synthetic entries (no disk IO)."""
    buf = TrajectoryBuffer.__new__(TrajectoryBuffer)
    buf.entries = [make_dummy_entry(**kwargs) for _ in range(n_entries)]
    return buf


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

class TestAsymmetricReward:
    def test_zero_action_zero_reward(self):
        """Flat position → zero PnL, zero TC."""
        r = AsymmetricReward(tc_rate=0.001)
        reward, info = r.compute(
            action=0.0, prev_action=0.0,
            close_current=100.0, close_next=105.0,
            sigma_close=0.02,
        )
        assert reward == 0.0
        assert info.raw_pnl == 0.0
        assert info.tc_penalty == 0.0

    def test_long_positive_return(self):
        """Full long + positive return → positive reward."""
        r = AsymmetricReward(tc_rate=0.0)
        reward, info = r.compute(
            action=1.0, prev_action=1.0,
            close_current=100.0, close_next=105.0,
            sigma_close=0.02,
        )
        # return = 5/100 = 0.05, reward = 1.0 * 0.05 / 0.02 = 2.5
        assert reward == pytest.approx(2.5, abs=1e-6)
        assert info.tc_penalty == 0.0

    def test_short_positive_return(self):
        """Full short + positive return → negative reward."""
        r = AsymmetricReward(tc_rate=0.0)
        reward, info = r.compute(
            action=-1.0, prev_action=-1.0,
            close_current=100.0, close_next=105.0,
            sigma_close=0.02,
        )
        assert reward == pytest.approx(-2.5, abs=1e-6)

    def test_transaction_cost(self):
        """Position change incurs TC penalty."""
        r = AsymmetricReward(tc_rate=0.001)
        reward, info = r.compute(
            action=1.0, prev_action=-1.0,
            close_current=100.0, close_next=100.0,
            sigma_close=0.02,
        )
        # return = 0, raw_pnl = 0, tc = 0.001 * |1 - (-1)| = 0.002
        assert info.tc_penalty == pytest.approx(0.002, abs=1e-8)
        assert reward == pytest.approx(-0.002, abs=1e-8)

    def test_high_vol_dampens_reward(self):
        """Higher sigma_close → smaller reward magnitude."""
        r = AsymmetricReward(tc_rate=0.0)
        _, info_lo = r.compute(
            action=1.0, prev_action=1.0,
            close_current=100.0, close_next=105.0,
            sigma_close=0.01,
        )
        _, info_hi = r.compute(
            action=1.0, prev_action=1.0,
            close_current=100.0, close_next=105.0,
            sigma_close=0.10,
        )
        # Higher vol → lower raw PnL
        assert abs(info_lo.raw_pnl) > abs(info_hi.raw_pnl)


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestLatentCryptoEnv:
    @pytest.fixture
    def env(self):
        buf = make_buffer(n_entries=10)
        config = EnvConfig(obs_dim=415, n_tgt=8, tc_rate=0.001, patch_len=16)
        return LatentCryptoEnv(buffer=buf, config=config)

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (415,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        assert "realized_future_idx" in info

    def test_step_returns_valid(self, env):
        env.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (415,)
        assert isinstance(reward, float)
        assert not terminated  # First step, not done yet
        assert not truncated
        assert np.isfinite(reward)

    def test_episode_terminates_after_n_tgt(self, env):
        """Episode must terminate after exactly n_tgt steps."""
        env.reset(seed=42)
        for step in range(env.config.n_tgt):
            action = np.array([0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if step < env.config.n_tgt - 1:
                assert not terminated, f"Terminated too early at step {step}"
            else:
                assert terminated, f"Should terminate at step {step}"

    def test_action_clipping(self, env):
        """Actions outside [-1, 1] should be clipped."""
        env.reset(seed=42)
        action = np.array([5.0], dtype=np.float32)  # Out of bounds
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["position"] == 1.0  # Clipped to 1.0

        env.reset(seed=42)
        action = np.array([-3.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["position"] == -1.0  # Clipped to -1.0

    def test_cumulative_pnl_tracks(self, env):
        """Cumulative PnL should accumulate over steps."""
        env.reset(seed=42)
        total = 0.0
        for _ in range(4):
            action = np.array([1.0], dtype=np.float32)
            _, reward, _, _, info = env.step(action)
            total += reward
        assert info["cumulative_pnl"] == pytest.approx(total, abs=1e-6)

    def test_observation_finite(self, env):
        """All observations should be finite throughout an episode."""
        env.reset(seed=42)
        for _ in range(env.config.n_tgt):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert np.all(np.isfinite(obs)), f"Non-finite obs: {obs}"

    def test_reward_order_of_magnitude(self, env):
        """Rewards should be O(1) — not explosive."""
        env.reset(seed=42)
        rewards = []
        for _ in range(env.config.n_tgt):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        rewards = np.array(rewards)
        assert np.all(np.abs(rewards) < 1000), f"Explosive rewards: {rewards}"

    def test_multiple_resets(self, env):
        """Environment should handle multiple resets without issues."""
        for _ in range(5):
            obs, info = env.reset()
            assert obs.shape == (415,)
            action = np.array([0.0], dtype=np.float32)
            obs, _, _, _, _ = env.step(action)
            assert obs.shape == (415,)


# ---------------------------------------------------------------------------
# Buffer split tests
# ---------------------------------------------------------------------------

class TestBufferSplit:
    def test_split_sizes(self):
        """Split should produce correct train/eval sizes."""
        buf = make_buffer(n_entries=20)
        train_buf, eval_buf = buf.split(val_ratio=0.2, seed=42)
        assert len(train_buf) == 16
        assert len(eval_buf) == 4
        assert len(train_buf) + len(eval_buf) == 20

    def test_split_no_overlap(self):
        """Train and eval should contain disjoint entries."""
        buf = make_buffer(n_entries=10)
        train_buf, eval_buf = buf.split(val_ratio=0.3, seed=42)
        train_ids = {id(e) for e in train_buf.entries}
        eval_ids = {id(e) for e in eval_buf.entries}
        assert train_ids.isdisjoint(eval_ids)

    def test_split_deterministic(self):
        """Same seed should produce the same split."""
        buf = make_buffer(n_entries=10)
        t1, e1 = buf.split(val_ratio=0.2, seed=99)
        t2, e2 = buf.split(val_ratio=0.2, seed=99)
        assert len(t1) == len(t2)
        assert len(e1) == len(e2)
        for a, b in zip(t1.entries, t2.entries):
            assert id(a) == id(b)


# ---------------------------------------------------------------------------
# Gymnasium check_env compatibility
# ---------------------------------------------------------------------------

class TestGymnasiumCompat:
    def test_check_env(self):
        """gymnasium.utils.env_checker.check_env should pass."""
        from gymnasium.utils.env_checker import check_env

        buf = make_buffer(n_entries=5)
        config = EnvConfig(obs_dim=415, n_tgt=8, tc_rate=0.001, patch_len=16)
        env = LatentCryptoEnv(buffer=buf, config=config)
        # check_env raises on failure
        check_env(env, skip_render_check=True)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_config(self, tmp_path):
        """Config loads from YAML correctly."""
        yaml_content = """
env:
  obs_dim: 415
  n_tgt: 8
  tc_rate: 0.001
  patch_len: 16
buffer:
  buffer_dir: "data/trajectory_buffer/"
  n_episodes: 255
  n_futures: 16
  refresh_ratio: 0.2
  refresh_every_epochs: 10
  val_ratio: 0.2
ppo:
  lr: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  total_timesteps: 1000000
  eval_freq: 10000
  log_dir: "tb_logs/strate_iv/"
"""
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml_content)
        config = load_config(str(cfg_file))
        assert isinstance(config, StrateIVConfig)
        assert config.env.obs_dim == 415
        assert config.ppo.lr == pytest.approx(3e-4)
        assert config.buffer.n_episodes == 255
