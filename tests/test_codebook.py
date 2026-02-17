import pytest
import torch

from src.strate_i.codebook import SphericalCodebook
from src.common.math_utils import l2_normalize


@pytest.fixture
def cb():
    return SphericalCodebook(num_codes=64, latent_dim=16)


@pytest.fixture
def z_e():
    return l2_normalize(torch.randn(32, 16))


def test_output_shapes(cb, z_e):
    out = cb(z_e)
    assert out["z_q"].shape == (32, 16)
    assert out["indices"].shape == (32,)


def test_ste_gradient(cb):
    raw = torch.randn(32, 16, requires_grad=True)
    z = l2_normalize(raw)
    out = cb(z)
    assert out["z_q"].grad_fn is not None
    out["z_q"].sum().backward()
    assert raw.grad is not None


def test_embeddings_normalized(cb, z_e):
    cb(z_e)
    norms = torch.linalg.norm(cb.embeddings, dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


def test_initialization(cb, z_e):
    assert not cb.initialized.item()
    cb(z_e)
    assert cb.initialized.item()


def test_encode(cb, z_e):
    # Initialize codebook first
    cb(z_e)
    cb.eval()
    # In eval mode, both should give same indices (no EMA update)
    out = cb(z_e)
    indices = cb.encode(z_e)
    torch.testing.assert_close(out["indices"], indices)


def test_perplexity_range(cb, z_e):
    out = cb(z_e)
    assert out["perplexity"].item() > 0
    assert out["perplexity"].item() <= cb.num_codes
