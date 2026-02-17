"""Scale invariance criterion: same pattern at different price scales -> identical tokens."""

import pytest
import torch

from scripts.generate_synthetic_data import generate_head_and_shoulders
from src.strate_i.data.transforms import compute_log_returns, extract_patches
from src.strate_i.tokenizer import TopologicalTokenizer
from src.strate_i.config import (
    StrateIConfig,
    PatchConfig,
    EncoderConfig,
    DecoderConfig,
    CodebookConfig,
)


def _small_config() -> StrateIConfig:
    return StrateIConfig(
        patch=PatchConfig(patch_length=16, stride=16, n_channels=5),
        encoder=EncoderConfig(in_channels=5, hidden_channels=32, latent_dim=16, n_layers=2),
        decoder=DecoderConfig(latent_dim=16, hidden_channels=32, out_channels=5, patch_length=16, n_layers=2),
        codebook=CodebookConfig(num_codes=64, latent_dim=16),
    )


def _series_to_patches(series: torch.Tensor, patch_length: int = 16, stride: int = 16) -> torch.Tensor:
    """Raw OHLCV (T, 5) -> log-returns -> patches (N, L, C)."""
    lr = compute_log_returns(series)
    return extract_patches(lr, patch_length, stride)


@pytest.fixture
def tokenizer():
    config = _small_config()
    tok = TopologicalTokenizer(config)
    tok.eval()
    return tok


def test_scale_invariance_criterion(tokenizer):
    """Same H&S at $0.001 and $90,000 must produce identical token sequences."""
    torch.manual_seed(42)

    # Generate same pattern at two scales (deterministic with same seed for noise)
    torch.manual_seed(0)
    series_low = generate_head_and_shoulders(T=512, base_price=0.001, amplitude=0.0002)
    torch.manual_seed(0)
    series_high = generate_head_and_shoulders(T=512, base_price=90000.0, amplitude=18000.0)

    patches_low = _series_to_patches(series_low)
    patches_high = _series_to_patches(series_high)

    assert patches_low.shape == patches_high.shape
    assert patches_low.shape[0] > 0

    # Log-returns should be approximately equal (scale-invariant by design)
    torch.testing.assert_close(
        patches_low, patches_high, atol=1e-3, rtol=1e-3,
        msg="Log-returns of same relative pattern should be nearly identical"
    )

    # Tokenize both
    indices_low = tokenizer.tokenize(patches_low)
    indices_high = tokenizer.tokenize(patches_high)

    torch.testing.assert_close(
        indices_low, indices_high,
        msg="Token sequences for same pattern at different scales MUST be identical"
    )


def test_proportional_amplitude_invariance(tokenizer):
    """Same relative amplitude (10%) at different prices -> same tokens."""
    torch.manual_seed(0)
    series_a = generate_head_and_shoulders(T=512, base_price=500.0, amplitude=50.0)
    torch.manual_seed(0)
    series_b = generate_head_and_shoulders(T=512, base_price=0.2, amplitude=0.02)

    patches_a = _series_to_patches(series_a)
    patches_b = _series_to_patches(series_b)

    indices_a = tokenizer.tokenize(patches_a)
    indices_b = tokenizer.tokenize(patches_b)

    torch.testing.assert_close(
        indices_a, indices_b,
        msg="Token sequences for proportional amplitudes should be identical"
    )
