import pytest
import torch

from src.strate_i.spherical_vqvae import SphericalVQVAE
from src.strate_i.config import EncoderConfig, DecoderConfig, CodebookConfig


@pytest.fixture
def small_enc():
    return EncoderConfig(in_channels=5, hidden_channels=32, latent_dim=16, n_layers=2)


@pytest.fixture
def small_dec():
    return DecoderConfig(latent_dim=16, hidden_channels=32, out_channels=5, patch_length=16, n_layers=2)


@pytest.fixture
def small_cb():
    return CodebookConfig(num_codes=64, latent_dim=16)


@pytest.fixture
def vqvae(small_enc, small_dec, small_cb):
    return SphericalVQVAE(small_enc, small_dec, small_cb)


@pytest.fixture
def batch():
    return torch.randn(8, 16, 5)


def test_forward_shapes(vqvae, batch):
    out = vqvae(batch)
    assert out["x_hat"].shape == (8, 16, 5)
    assert out["z_e"].shape == (8, 16)
    assert out["z_q"].shape == (8, 16)
    assert out["indices"].shape == (8,)


def test_encode_returns_indices(vqvae, batch):
    indices = vqvae.encode(batch)
    assert indices.shape == (8,)
    assert indices.dtype == torch.long


def test_decode_from_indices(vqvae, batch):
    # Run forward first to initialize codebook
    vqvae(batch)
    indices = torch.randint(0, 64, (8,))
    x_hat = vqvae.decode_from_indices(indices)
    assert x_hat.shape == (8, 16, 5)


def test_end_to_end_finite(vqvae, batch):
    out = vqvae(batch)
    assert torch.all(torch.isfinite(out["x_hat"]))
    assert torch.all(torch.isfinite(out["z_e"]))
    assert torch.all(torch.isfinite(out["commitment_loss"]))
