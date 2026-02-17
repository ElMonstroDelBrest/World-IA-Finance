"""Stochastic MLP predictor for Fin-JEPA (Strate III).

Takes context encoder output at target positions, optional latent noise z,
and predicts the target encoder representations.

Backward-compatible: when z=None, behaves identically to the deterministic
Strate II predictor (z is treated as zeros).
"""

import torch
from torch import Tensor, nn


class Predictor(nn.Module):
    """Stochastic MLP predictor: maps context representations at target
    positions (+ optional noise z) to predicted target representations.

    Args:
        d_model: Dimension of encoder representations.
        hidden_dim: Hidden layer dimension.
        n_layers: Number of MLP layers.
        seq_len: Maximum sequence length (for positional embeddings).
        dropout: Dropout probability.
        z_dim: Dimension of latent noise vector (0 = deterministic).
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        seq_len: int = 64,
        dropout: float = 0.1,
        z_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim

        # Learnable positional embedding for target positions
        self.pos_embed = nn.Embedding(seq_len, d_model)

        # MLP: context_mean || pos_embed [|| z] → predicted representation
        in_dim = d_model * 2 + z_dim  # context_mean + pos_embed + z
        layers = []
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else d_model
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, h_x: Tensor, target_positions: Tensor, z: Tensor | None = None
    ) -> Tensor:
        """Predict target representations from context encoder output.

        Args:
            h_x: (B, S, d_model) context encoder output (full sequence).
            target_positions: (B, N_tgt) int64 indices of target positions.
            z: (B, N_tgt, z_dim) latent noise. None → deterministic (zeros).

        Returns:
            (B, N_tgt, d_model) predicted target representations.
        """
        B = h_x.shape[0]
        N_tgt = target_positions.shape[1]

        # Mean-pool the context representation
        ctx_mean = h_x.mean(dim=1, keepdim=True)  # (B, 1, d_model)

        # Get positional embeddings for target positions
        pos = self.pos_embed(target_positions)  # (B, N_tgt, d_model)

        # Expand context mean to match target positions
        ctx_expanded = ctx_mean.expand(-1, N_tgt, -1)  # (B, N_tgt, d_model)

        # Concatenate context + position
        parts = [ctx_expanded, pos]

        # Add noise z if provided, otherwise zeros
        if self.z_dim > 0:
            if z is None:
                z = torch.zeros(B, N_tgt, self.z_dim, device=h_x.device, dtype=h_x.dtype)
            parts.append(z)

        x = torch.cat(parts, dim=-1)  # (B, N_tgt, 2*d_model + z_dim)
        return self.mlp(x)  # (B, N_tgt, d_model)
