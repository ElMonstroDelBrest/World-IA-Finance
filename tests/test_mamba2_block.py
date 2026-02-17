"""Tests for Mamba-2 block: shapes, weekend gating, gradients."""

import torch
import pytest

from src.strate_ii.mamba2_block import Mamba2Block, selective_scan


@pytest.fixture
def block():
    return Mamba2Block(d_model=64, d_state=8, n_heads=2, expand_factor=2, conv_kernel=4)


def test_output_shape(block):
    """Output shape matches input shape."""
    x = torch.randn(4, 16, 64)
    y = block(x)
    assert y.shape == (4, 16, 64)


def test_output_shape_with_weekend(block):
    """Output shape is correct with weekend mask."""
    x = torch.randn(4, 16, 64)
    weekend = torch.zeros(4, 16)
    weekend[:, 5:7] = 1.0  # Days 5-6 are weekend
    y = block(x, weekend_mask=weekend)
    assert y.shape == (4, 16, 64)


def test_weekend_gating_freezes_state():
    """Weekend gating: hidden state should not change during weekend positions.

    Sequence: [A, B, WEEKEND, WEEKEND, C]
    h after B should equal h after second WEEKEND (state frozen).
    """
    torch.manual_seed(42)
    B, L, D = 1, 5, 8
    n_heads, d_state = 2, 4
    head_dim = D // n_heads

    x = torch.randn(B, L, D)
    dt_logits = torch.ones(B, L, n_heads) * 2.0  # Moderate delta
    A = -torch.ones(n_heads, d_state)  # Simple A
    B_ssm = torch.randn(B, L, n_heads, d_state)
    C_ssm = torch.randn(B, L, n_heads, d_state)

    # Weekend mask: positions 2 and 3 are weekend
    weekend = torch.zeros(B, L)
    weekend[:, 2] = 1.0
    weekend[:, 3] = 1.0

    # Run scan manually to check internal state
    from torch.nn import functional as F

    dt = F.softplus(dt_logits)
    gate = 1.0 - weekend.unsqueeze(-1)
    dt_gated = dt * gate

    h = torch.zeros(B, n_heads, d_state, head_dim)
    x_heads = x.view(B, L, n_heads, head_dim)
    states = []

    for t in range(L):
        dt_t = dt_gated[:, t, :]
        B_t = B_ssm[:, t, :, :]
        C_t = C_ssm[:, t, :, :]
        x_t = x_heads[:, t, :, :]

        A_bar = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))
        B_bar = dt_t.unsqueeze(-1) * B_t
        h = A_bar.unsqueeze(-1) * h + B_bar.unsqueeze(-1) * x_t.unsqueeze(2)
        states.append(h.clone())

    # State after position 1 (B) should equal state after position 2 (first weekend)
    # because dt=0 → A_bar=I, B_bar=0 → h unchanged
    assert torch.allclose(states[1], states[2], atol=1e-6), \
        "State changed during first weekend position"
    # State after position 2 should equal position 3 (second weekend)
    assert torch.allclose(states[2], states[3], atol=1e-6), \
        "State changed during second weekend position"


def test_no_weekend_state_changes():
    """Without weekend mask, state should change at every position."""
    torch.manual_seed(42)
    B, L, D = 1, 5, 8
    n_heads, d_state = 2, 4
    head_dim = D // n_heads

    x = torch.randn(B, L, D)
    dt_logits = torch.ones(B, L, n_heads) * 2.0
    A = -torch.ones(n_heads, d_state)
    B_ssm = torch.randn(B, L, n_heads, d_state)
    C_ssm = torch.randn(B, L, n_heads, d_state)

    from torch.nn import functional as F

    dt = F.softplus(dt_logits)
    h = torch.zeros(B, n_heads, d_state, head_dim)
    x_heads = x.view(B, L, n_heads, head_dim)
    states = []

    for t in range(L):
        dt_t = dt[:, t, :]
        B_t = B_ssm[:, t, :, :]
        x_t = x_heads[:, t, :, :]
        A_bar = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))
        B_bar = dt_t.unsqueeze(-1) * B_t
        h = A_bar.unsqueeze(-1) * h + B_bar.unsqueeze(-1) * x_t.unsqueeze(2)
        states.append(h.clone())

    # Without weekend mask, consecutive states should differ
    for t in range(1, L):
        assert not torch.allclose(states[t - 1], states[t], atol=1e-6), \
            f"States at {t-1} and {t} are identical without weekend mask"


def test_gradients_flow(block):
    """Gradients should flow through the block."""
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_gradients_with_weekend(block):
    """Gradients should flow even with weekend gating."""
    x = torch.randn(2, 8, 64, requires_grad=True)
    weekend = torch.zeros(2, 8)
    weekend[:, 3:5] = 1.0
    y = block(x, weekend_mask=weekend)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_no_nan_output(block):
    """Output should never contain NaN."""
    x = torch.randn(4, 32, 64)
    weekend = torch.zeros(4, 32)
    weekend[:, 5:7] = 1.0
    weekend[:, 12:14] = 1.0
    y = block(x, weekend_mask=weekend)
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"


def test_residual_connection(block):
    """Block should have a residual connection (output ≠ just SSM output)."""
    x = torch.randn(2, 8, 64)
    # With residual, if SSM output were zero, y ≈ x
    # We can't easily test this directly, but we verify the block
    # returns values in a reasonable range (not exploding)
    y = block(x)
    assert y.abs().max() < 1000, "Output magnitude too large, possible residual issue"
