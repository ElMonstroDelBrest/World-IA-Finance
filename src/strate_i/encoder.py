"""Causal dilated convolutional encoder mapping patches to the unit sphere."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.common.math_utils import l2_normalize


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x + residual


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        n_layers: int = 4,
        dilation_base: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.layers = nn.ModuleList([
            CausalResidualBlock(hidden_channels, kernel_size, dilation_base**i)
            for i in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """(B, L, C) -> (B, D). Output is L2-normalized (on the sphere)."""
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        z = x[:, :, -1]  # last timestep (causal)
        z = self.output_proj(z)
        return l2_normalize(z, dim=-1)
