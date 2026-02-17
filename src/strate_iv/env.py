"""LatentCryptoEnv: Gymnasium environment for Strate IV RL training.

The agent observes a 416-dim vector composed of:
  - h_x_pooled (128): JEPA context encoder mean-pool
  - future_latent_mean (128): Mean of N future latents across N samples
  - future_latent_std (128): Std of N future latents across N samples
  - future_close_stats (24): Per-target close return stats (8 targets * 3: mean/std/skew)
  - revin_stds (5): RevIN std per channel (volatility regime)
  - delta_mu (1): Macro trend — normalized mu variation between last context patches
  - position (1): Current portfolio position a_{t-1}
  - cumulative_pnl (1): Running PnL

Action: Box([-1], [1]) — continuous position (-1=short, 0=flat, +1=long).

Episode: N_tgt=8 steps. At reset, one future is sampled as "realized" (domain randomization).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward import AsymmetricReward
from .trajectory_buffer import TrajectoryBuffer, TrajectoryEntry
from .config import EnvConfig


class LatentCryptoEnv(gym.Env):
    """Latent-space crypto trading environment for PPO training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        buffer: TrajectoryBuffer,
        config: EnvConfig | None = None,
    ):
        super().__init__()
        self.buffer = buffer
        self.config = config or EnvConfig()
        self.reward_fn = AsymmetricReward(tc_rate=self.config.tc_rate)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Episode state
        self._entry: TrajectoryEntry | None = None
        self._realized_idx: int = 0
        self._step_idx: int = 0
        self._position: float = 0.0
        self._cumulative_pnl: float = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Use np_random for deterministic sampling when seeded
        entry_idx = self.np_random.integers(0, len(self.buffer))
        self._entry = self.buffer.entries[entry_idx]

        # Domain randomization: pick one future as "realized"
        n_futures = self._entry.future_ohlcv.shape[0]
        self._realized_idx = self.np_random.integers(0, n_futures)

        self._step_idx = 0
        self._position = 0.0
        self._cumulative_pnl = 0.0

        obs = self._build_observation()
        info = {"realized_future_idx": self._realized_idx}
        return obs, info

    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))

        # Get close prices for current and next step from realized future
        # future_ohlcv: (N, N_tgt, patch_len, 5)
        realized = self._entry.future_ohlcv[self._realized_idx]  # (N_tgt, patch_len, 5)

        # Close price = channel 3, last candle of each patch
        close_current = realized[self._step_idx, -1, 3].item()

        if self._step_idx < self.config.n_tgt - 1:
            close_next = realized[self._step_idx + 1, -1, 3].item()
        else:
            # Last step: use last candle close of current patch
            close_next = close_current

        reward, info = self.reward_fn.compute(
            action=action_val,
            prev_action=self._position,
            close_current=close_current,
            close_next=close_next,
        )

        self._position = action_val
        self._cumulative_pnl += reward
        self._step_idx += 1

        terminated = self._step_idx >= self.config.n_tgt
        truncated = False

        obs = self._build_observation()

        step_info = {
            "raw_pnl": info.raw_pnl,
            "tc_penalty": info.tc_penalty,
            "log_return": info.log_return,
            "position": self._position,
            "cumulative_pnl": self._cumulative_pnl,
            "step": self._step_idx,
        }

        return obs, reward, terminated, truncated, step_info

    def _build_observation(self) -> np.ndarray:
        """Build the 416-dim observation vector."""
        entry = self._entry
        future_latents = entry.future_latents.numpy()  # (N, N_tgt, d_model)
        N, N_tgt, d_model = future_latents.shape

        # 1. h_x_pooled (128)
        h_x_pooled = entry.h_x_pooled.numpy()  # (d_model,)

        # 2. future_latent_mean (128) — mean across N futures, mean-pool across N_tgt
        future_mean = future_latents.mean(axis=0).mean(axis=0)  # (d_model,)

        # 3. future_latent_std (128) — std across N futures, mean-pool across N_tgt
        future_std = future_latents.std(axis=0).mean(axis=0)  # (d_model,)

        # 4. future_close_stats (24) — per-target: mean, std, skew of close returns
        #    future_ohlcv: (N, N_tgt, patch_len, 5)
        future_ohlcv = entry.future_ohlcv.numpy()
        close_stats = self._compute_close_stats(future_ohlcv)  # (24,)

        # 5. revin_stds (5)
        revin_stds = entry.revin_stds.numpy().flatten()  # (5,)

        # 6. delta_mu (1) — macro trend from context OHLCV
        delta_mu = self._compute_delta_mu(entry)  # (1,)

        # 7. position (1)
        position = np.array([self._position], dtype=np.float32)

        # 8. cumulative pnl (1)
        cum_pnl = np.array([self._cumulative_pnl], dtype=np.float32)

        obs = np.concatenate([
            h_x_pooled, future_mean, future_std,
            close_stats, revin_stds, delta_mu,
            position, cum_pnl,
        ]).astype(np.float32)

        # Replace any NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    @staticmethod
    def _compute_delta_mu(entry: TrajectoryEntry) -> np.ndarray:
        """Compute macro trend signal from context OHLCV.

        Computes the normalized difference between the mean close of the
        last 2 patches vs the previous 2 patches in the context window.
        This gives the agent the "slope" of the global trend.

        Returns:
            (1,) array with normalized delta_mu.
        """
        context = entry.context_ohlcv.numpy()  # (T, 5)
        close = context[:, 3]  # (T,)
        T = len(close)
        patch_len = 16

        if T < 4 * patch_len:
            return np.zeros(1, dtype=np.float32)

        # Mean close of last 2 patches
        recent = close[-(2 * patch_len):].mean()
        # Mean close of the 2 patches before that
        earlier = close[-(4 * patch_len):-(2 * patch_len)].mean()

        # Normalize by overall std to keep it O(1)
        sigma = close.std() + 1e-8
        delta_mu = (recent - earlier) / sigma

        return np.array([delta_mu], dtype=np.float32)

    @staticmethod
    def _compute_close_stats(future_ohlcv: np.ndarray) -> np.ndarray:
        """Compute per-target close return statistics across N futures.

        Args:
            future_ohlcv: (N, N_tgt, patch_len, 5)

        Returns:
            (N_tgt * 3,) = (24,) for N_tgt=8: [mean, std, skew] per target.
        """
        N, N_tgt, patch_len, _ = future_ohlcv.shape

        # Close channel = 3, last candle of each patch
        close_prices = future_ohlcv[:, :, -1, 3]  # (N, N_tgt)

        # Returns: ratio of close at target t vs target t-1
        eps = 1e-8
        close_shifted = np.concatenate([
            close_prices[:, :1],  # anchor
            close_prices[:, :-1],
        ], axis=1)  # (N, N_tgt)
        returns = (close_prices - close_shifted) / (np.abs(close_shifted) + eps)

        stats = []
        for t in range(N_tgt):
            r = returns[:, t]  # (N,)
            mean = r.mean()
            std = r.std() + eps
            skew = ((r - mean) ** 3).mean() / (std ** 3)
            stats.extend([mean, std, skew])

        return np.array(stats, dtype=np.float32)
