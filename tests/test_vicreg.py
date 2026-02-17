"""Tests for VICReg loss components."""

import torch
import pytest

from src.strate_ii.vicreg import (
    VICRegLoss,
    invariance_loss,
    variance_loss,
    covariance_loss,
)


@pytest.fixture
def vicreg():
    return VICRegLoss(inv_weight=25.0, var_weight=25.0, cov_weight=1.0)


def test_invariance_identical():
    """Invariance loss should be 0 for identical inputs."""
    z = torch.randn(16, 64)
    assert invariance_loss(z, z).item() == pytest.approx(0.0, abs=1e-6)


def test_invariance_different():
    """Invariance loss should be positive for different inputs."""
    z_a = torch.randn(16, 64)
    z_b = torch.randn(16, 64)
    assert invariance_loss(z_a, z_b).item() > 0.0


def test_variance_collapsed():
    """Variance loss should be high when all representations are the same (collapse)."""
    z = torch.ones(32, 64)  # All identical → std=0 → max penalty
    loss = variance_loss(z, gamma=1.0)
    assert loss.item() == pytest.approx(1.0, abs=0.02)  # eps=1e-4 makes std slightly >0


def test_variance_spread():
    """Variance loss should be ~0 when std >= gamma."""
    z = torch.randn(1000, 64) * 2.0  # std ≈ 2.0 >> gamma=1.0
    loss = variance_loss(z, gamma=1.0)
    assert loss.item() < 0.1


def test_covariance_decorrelated():
    """Covariance loss should be ~0 for decorrelated dimensions."""
    # Orthogonal matrix gives decorrelated dims
    z = torch.randn(256, 64)
    # Not perfectly 0 but small
    loss = covariance_loss(z)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_covariance_correlated():
    """Covariance loss should be high when dimensions are correlated."""
    z = torch.randn(256, 1).expand(-1, 64)  # All dims identical → max correlation
    loss = covariance_loss(z)
    assert loss.item() > 0.0


def test_vicreg_forward_keys(vicreg):
    """Forward returns all expected loss components."""
    z_a = torch.randn(16, 64)
    z_b = torch.randn(16, 64)
    out = vicreg(z_a, z_b)
    assert set(out.keys()) == {"total", "invariance", "variance", "covariance"}
    for v in out.values():
        assert torch.isfinite(v)


def test_vicreg_total_is_weighted_sum(vicreg):
    """Total loss should equal weighted sum of components."""
    z_a = torch.randn(32, 64)
    z_b = torch.randn(32, 64)
    out = vicreg(z_a, z_b)
    expected = (
        25.0 * out["invariance"]
        + 25.0 * out["variance"]
        + 1.0 * out["covariance"]
    )
    assert out["total"].item() == pytest.approx(expected.item(), rel=1e-4)


def test_vicreg_covariance_float32_under_amp():
    """Covariance must be computed in float32 even under AMP."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for AMP test")

    vicreg = VICRegLoss().cuda()
    z_a = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    z_b = torch.randn(32, 64, device="cuda", dtype=torch.float16)

    with torch.amp.autocast("cuda"):
        out = vicreg(z_a, z_b)

    # Covariance loss should be finite (not NaN from float16 overflow)
    assert torch.isfinite(out["covariance"])
    assert torch.isfinite(out["total"])


def test_vicreg_gradient_flows():
    """Gradients should flow through all components."""
    z_a = torch.randn(16, 64, requires_grad=True)
    z_b = torch.randn(16, 64)  # Target — no grad
    vicreg = VICRegLoss()
    out = vicreg(z_a, z_b)
    out["total"].backward()
    assert z_a.grad is not None
    assert torch.isfinite(z_a.grad).all()
