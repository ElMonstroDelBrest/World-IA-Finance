"""Mamba-2 block with selective scan and weekend gating.

Custom pure-PyTorch implementation (no mamba-ssm dependency).
Sequences are short (~64 tokens) so sequential scan is sufficient.

Weekend gating: delta *= (1 - is_weekend)
  When weekend: delta=0 → exp(A*0)=I → h_t = h_{t-1} (state frozen).
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left padding."""

    def __init__(self, d_inner: int, kernel_size: int = 4):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size, groups=d_inner)

    def forward(self, x: Tensor) -> Tensor:
        """(B, D, L) -> (B, D, L)."""
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


def selective_scan(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    weekend_mask: Tensor | None = None,
) -> Tensor:
    """Sequential selective scan with weekend gating.

    Args:
        x: (B, L, D_inner) input after conv+activation.
        dt: (B, L, n_heads) delta (time step) logits, pre-softplus.
        A: (n_heads, d_state) diagonal SSM matrix (log-space, negative).
        B: (B, L, n_heads, d_state) input-dependent SSM input matrix.
        C: (B, L, n_heads, d_state) input-dependent SSM output matrix.
        weekend_mask: (B, L) float {0.0, 1.0} where 1.0 = weekend.

    Returns:
        y: (B, L, D_inner) scan output.
    """
    B_batch, L, D_inner = x.shape
    n_heads = A.shape[0]
    d_state = A.shape[1]
    head_dim = D_inner // n_heads

    # Reshape x into heads: (B, L, n_heads, head_dim)
    x_heads = x.view(B_batch, L, n_heads, head_dim)

    # Apply softplus to dt logits
    dt = F.softplus(dt)  # (B, L, n_heads)

    # Weekend gating: zero out delta on weekends
    if weekend_mask is not None:
        # weekend_mask: (B, L) → (B, L, 1)
        gate = 1.0 - weekend_mask.unsqueeze(-1)
        dt = dt * gate

    # Initialize hidden state: (B, n_heads, d_state, head_dim)
    h = torch.zeros(B_batch, n_heads, d_state, head_dim, device=x.device, dtype=x.dtype)
    outputs = []

    for t in range(L):
        dt_t = dt[:, t, :]  # (B, n_heads)
        B_t = B[:, t, :, :]  # (B, n_heads, d_state)
        C_t = C[:, t, :, :]  # (B, n_heads, d_state)
        x_t = x_heads[:, t, :, :]  # (B, n_heads, head_dim)

        # Discretize: A_bar = exp(A * dt), B_bar = dt * B
        # A is (n_heads, d_state), dt_t is (B, n_heads)
        A_bar = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (B, n_heads, d_state)
        B_bar = dt_t.unsqueeze(-1) * B_t  # (B, n_heads, d_state)

        # State update: h_t = A_bar * h_{t-1} + B_bar * x_t
        # h: (B, n_heads, d_state, head_dim)
        # A_bar: (B, n_heads, d_state) → unsqueeze for head_dim
        # B_bar: (B, n_heads, d_state) → outer product with x_t
        h = A_bar.unsqueeze(-1) * h + B_bar.unsqueeze(-1) * x_t.unsqueeze(2)

        # Output: y_t = C_t @ h_t
        # C_t: (B, n_heads, d_state), h: (B, n_heads, d_state, head_dim)
        y_t = torch.einsum("bns,bnsd->bnd", C_t, h)  # (B, n_heads, head_dim)
        outputs.append(y_t)

    # Stack and reshape: (B, L, n_heads, head_dim) → (B, L, D_inner)
    y = torch.stack(outputs, dim=1)
    y = y.reshape(B_batch, L, D_inner)
    return y


class Mamba2Block(nn.Module):
    """Single Mamba-2 block with selective scan and weekend gating.

    Architecture:
        x → LayerNorm → Linear → split(x_branch, z_gate, B, C, dt)
        x_branch → CausalConv1d → SiLU → selective_scan → y
        y = y * SiLU(z_gate)
        y → Linear → + residual
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        n_heads: int = 2,
        expand_factor: int = 2,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        self.d_state = d_state
        self.n_heads = n_heads
        self.head_dim = self.d_inner // n_heads

        assert self.d_inner % n_heads == 0, "d_inner must be divisible by n_heads"

        self.norm = nn.LayerNorm(d_model)

        # Input projection: x → x_branch, z_gate, B, C, dt
        # Sizes: d_inner + d_inner + n_heads*d_state + n_heads*d_state + n_heads
        self.in_proj_size = (
            self.d_inner        # x_branch
            + self.d_inner      # z_gate
            + n_heads * d_state # B
            + n_heads * d_state # C
            + n_heads           # dt
        )
        self.in_proj = nn.Linear(d_model, self.in_proj_size, bias=False)

        # Causal conv on x_branch
        self.conv = CausalConv1d(self.d_inner, conv_kernel)

        # SSM parameter A (log-space, negative, learnable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(n_heads, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor, weekend_mask: Tensor | None = None) -> Tensor:
        """Forward pass with optional weekend gating.

        Args:
            x: (B, L, d_model) input sequence.
            weekend_mask: (B, L) float {0.0, 1.0} where 1.0 = weekend.

        Returns:
            (B, L, d_model) output with residual connection.
        """
        residual = x
        x = self.norm(x)

        # Project and split
        proj = self.in_proj(x)  # (B, L, in_proj_size)

        idx = 0
        x_branch = proj[..., idx:idx + self.d_inner]
        idx += self.d_inner
        z_gate = proj[..., idx:idx + self.d_inner]
        idx += self.d_inner
        B_proj = proj[..., idx:idx + self.n_heads * self.d_state]
        idx += self.n_heads * self.d_state
        C_proj = proj[..., idx:idx + self.n_heads * self.d_state]
        idx += self.n_heads * self.d_state
        dt = proj[..., idx:idx + self.n_heads]

        # Reshape B, C: (B, L, n_heads * d_state) → (B, L, n_heads, d_state)
        B_proj = B_proj.view(*B_proj.shape[:-1], self.n_heads, self.d_state)
        C_proj = C_proj.view(*C_proj.shape[:-1], self.n_heads, self.d_state)

        # Causal conv on x_branch: (B, L, D) → (B, D, L) → conv → (B, L, D)
        x_branch = self.conv(x_branch.transpose(1, 2)).transpose(1, 2)
        x_branch = F.silu(x_branch)

        # SSM: A in negative log-space
        A = -torch.exp(self.A_log)  # (n_heads, d_state)

        # Selective scan with weekend gating
        y = selective_scan(x_branch, dt, A, B_proj, C_proj, weekend_mask)

        # Gated output
        y = y * F.silu(z_gate)

        # Output projection + residual
        return self.out_proj(y) + residual
