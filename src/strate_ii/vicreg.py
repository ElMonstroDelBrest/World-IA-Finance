"""VICReg loss: Variance-Invariance-Covariance Regularization (Bardes et al. 2022).

CRITICAL: Covariance computed in float32 even under AMP to ensure proper
orthogonalization of Momentum vs Volatility dimensions.
"""

import torch
from torch import Tensor


def invariance_loss(z_a: Tensor, z_b: Tensor) -> Tensor:
    """MSE between paired representations. (B, D) -> scalar."""
    return torch.nn.functional.mse_loss(z_a, z_b)


def variance_loss(z: Tensor, gamma: float = 1.0) -> Tensor:
    """Hinge loss on per-dimension std to prevent collapse. (B, D) -> scalar."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.nn.functional.relu(gamma - std).mean()


def covariance_loss(z: Tensor) -> Tensor:
    """Off-diagonal covariance penalty. (B, D) -> scalar.

    MUST be called in float32 context (even under AMP).
    """
    n, d = z.shape
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (n - 1)
    # Zero out diagonal, sum squares of off-diagonal
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / d


class VICRegLoss(torch.nn.Module):
    """VICReg loss combining invariance, variance, and covariance terms.

    Args:
        inv_weight: Weight for invariance (MSE) term.
        var_weight: Weight for variance (hinge) term.
        cov_weight: Weight for covariance (decorrelation) term.
        var_gamma: Target std for variance hinge.
    """

    def __init__(
        self,
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_gamma: float = 1.0,
    ):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_gamma = var_gamma

    def forward(self, z_a: Tensor, z_b: Tensor) -> dict[str, Tensor]:
        """Compute VICReg loss.

        Args:
            z_a: Predicted representations (B, D).
            z_b: Target representations (B, D). Detached (no grad).

        Returns:
            dict with total, invariance, variance, covariance losses.
        """
        inv = invariance_loss(z_a, z_b)

        # Covariance and variance MUST be in float32 for numerical stability
        with torch.amp.autocast("cuda", enabled=False):
            z_a_f = z_a.float()
            z_b_f = z_b.float()
            var = variance_loss(z_a_f, self.var_gamma) + variance_loss(z_b_f, self.var_gamma)
            cov = covariance_loss(z_a_f) + covariance_loss(z_b_f)

        total = self.inv_weight * inv + self.var_weight * var + self.cov_weight * cov

        return {
            "total": total,
            "invariance": inv,
            "variance": var,
            "covariance": cov,
        }
