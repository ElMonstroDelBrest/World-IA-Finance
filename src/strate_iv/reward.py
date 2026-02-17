"""Asymmetric reward with RevIN volatility scaling and transaction costs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RewardInfo:
    raw_pnl: float
    tc_penalty: float
    sigma_close: float


class AsymmetricReward:
    """Reward = position * return / sigma_close - tc_rate * |delta_position|.

    The division by sigma_close (RevIN std of Close channel) punishes
    leverage in high-volatility regimes — the agent earns less reward
    per unit of return when vol is high.
    """

    def __init__(self, tc_rate: float = 0.001):
        self.tc_rate = tc_rate

    def compute(
        self,
        action: float,
        prev_action: float,
        close_current: float,
        close_next: float,
        sigma_close: float,
    ) -> tuple[float, RewardInfo]:
        """Compute asymmetric reward for a single step.

        Args:
            action: Current position a_t in [-1, 1].
            prev_action: Previous position a_{t-1} in [-1, 1].
            close_current: Close price at current step.
            close_next: Close price at next step.
            sigma_close: RevIN std of the Close channel (from context).

        Returns:
            (reward, info) where reward is the scalar reward and info
            contains the decomposition.
        """
        sigma_close = max(sigma_close, 1e-8)
        ret = (close_next - close_current) / max(abs(close_current), 1e-8)
        raw_pnl = action * ret / sigma_close
        tc_penalty = self.tc_rate * abs(action - prev_action)
        reward = raw_pnl - tc_penalty
        return reward, RewardInfo(
            raw_pnl=raw_pnl,
            tc_penalty=tc_penalty,
            sigma_close=sigma_close,
        )
