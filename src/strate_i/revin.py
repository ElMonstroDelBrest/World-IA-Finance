"""Reversible Instance Normalization (RevIN) for distribution shift handling."""

import torch
from torch import Tensor, nn


class RevIN(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, n_channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, n_channels))

    def normalize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Normalize per instance. Input: (B, L, C). Returns: (normalized, means, stds)."""
        means = x.mean(dim=1, keepdim=True)
        stds = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        x = (x - means) / stds
        if self.affine:
            x = x * self.weight + self.bias
        return x, means, stds

    def denormalize(self, x: Tensor, means: Tensor, stds: Tensor) -> Tensor:
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps)
        return x * stds + means


class StatsStore:
    """Plain Python dict to store per-patch (mean, std) for Strate IV access."""

    def __init__(self):
        self._stats: dict[str, tuple[Tensor, Tensor]] = {}

    def store(self, patch_id: str, mean: Tensor, std: Tensor):
        self._stats[patch_id] = (mean.detach().cpu(), std.detach().cpu())

    def get(self, patch_id: str) -> tuple[Tensor, Tensor]:
        return self._stats[patch_id]

    def clear(self):
        self._stats.clear()

    def __len__(self) -> int:
        return len(self._stats)
