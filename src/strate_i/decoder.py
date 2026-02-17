from torch import Tensor, nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)

    def forward(self, x: Tensor) -> Tensor:
        return x + F.gelu(self.conv2(F.gelu(self.conv1(x))))


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_channels: int = 128,
        out_channels: int = 5,
        patch_length: int = 16,
        n_layers: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.input_proj = nn.Linear(latent_dim, hidden_channels)
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_channels, kernel_size) for _ in range(n_layers)
        ])
        self.output_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, z: Tensor) -> Tensor:
        """(B, D) -> (B, L, C)."""
        x = self.input_proj(z)  # (B, H)
        x = x.unsqueeze(-1).expand(-1, -1, self.patch_length)  # (B, H, L)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)  # (B, C, L)
        return x.permute(0, 2, 1)  # (B, L, C)
