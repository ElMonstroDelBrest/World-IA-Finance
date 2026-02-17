"""Loss functions for Strate I: Huber reconstruction + Soft-DTW + VQ commitment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tslearn.metrics import SoftDTWLossPyTorch


def soft_dtw_loss(x: Tensor, x_hat: Tensor, gamma: float = 0.1) -> Tensor:
    """Soft-DTW loss between batched time series. (B, L, C) -> scalar."""
    sdtw_fn = SoftDTWLossPyTorch(gamma=gamma)
    return sdtw_fn(x, x_hat).mean()


class VQVAELoss(nn.Module):
    def __init__(
        self,
        huber_delta: float = 1.0,
        sdtw_alpha: float = 0.1,
        sdtw_gamma: float = 0.1,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.sdtw_alpha = sdtw_alpha
        self.sdtw_gamma = sdtw_gamma

    def forward(
        self, x: Tensor, x_hat: Tensor, commitment_loss: Tensor
    ) -> dict[str, Tensor]:
        """Compute total loss = huber + alpha*sdtw + commitment (already weighted by codebook)."""
        huber = F.huber_loss(x_hat, x, delta=self.huber_delta)
        sdtw = soft_dtw_loss(x, x_hat, gamma=self.sdtw_gamma) * self.sdtw_alpha
        total = huber + sdtw + commitment_loss
        return {
            "total": total,
            "huber": huber,
            "sdtw": sdtw,
            "commitment": commitment_loss,
        }
