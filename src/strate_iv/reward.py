"""Simplified PnL reward with transaction costs (no volatility scaling)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardInfo:
    raw_pnl: float
    tc_penalty: float
    log_return: float


class AsymmetricReward:
    """Reward = action * log_return - cost * |delta_action|.

    No sigma_close division — the agent sees raw market dynamics and can
    learn to follow trends / breakouts without being penalized by high vol.
    """

    def __init__(self, tc_rate: float = 0.0005):
        self.tc_rate = tc_rate

    def compute(
        self,
        action: float,
        prev_action: float,
        close_current: float,
        close_next: float,
        sigma_close: float = 0.0,  # kept for API compat, unused
    ) -> tuple[float, RewardInfo]:
        """Compute reward for a single step.

        Args:
            action: Current position a_t in [-1, 1].
            prev_action: Previous position a_{t-1} in [-1, 1].
            close_current: Close price at current step.
            close_next: Close price at next step.
            sigma_close: Unused (kept for backward compatibility).

        Returns:
            (reward, info)
        """
        eps = 1e-8
        log_ret = math.log(max(close_next, eps) / max(close_current, eps))
        raw_pnl = action * log_ret
        tc_penalty = self.tc_rate * abs(action - prev_action)
        reward = raw_pnl - tc_penalty
        return reward, RewardInfo(
            raw_pnl=raw_pnl,
            tc_penalty=tc_penalty,
            log_return=log_ret,
        )
