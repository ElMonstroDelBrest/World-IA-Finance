import pytest
import torch

from src.strate_i.revin import RevIN, StatsStore


@pytest.fixture
def data():
    return torch.randn(32, 16, 5)


def test_normalize_shape(data):
    revin = RevIN(n_channels=5)
    x_norm, means, stds = revin.normalize(data)
    assert x_norm.shape == data.shape
    assert means.shape == (32, 1, 5)
    assert stds.shape == (32, 1, 5)


def test_denormalize_inverts(data):
    revin = RevIN(n_channels=5, affine=False)
    x_norm, means, stds = revin.normalize(data)
    x_back = revin.denormalize(x_norm, means, stds)
    torch.testing.assert_close(x_back, data, atol=1e-5, rtol=1e-5)


def test_affine_params():
    revin = RevIN(n_channels=5, affine=True)
    assert hasattr(revin, "weight")
    assert hasattr(revin, "bias")
    assert revin.weight.shape == (1, 1, 5)

    revin_no = RevIN(n_channels=5, affine=False)
    assert not hasattr(revin_no, "weight")


def test_stats_store():
    store = StatsStore()
    mean = torch.randn(1, 5)
    std = torch.rand(1, 5)
    store.store("patch_0", mean, std)
    m, s = store.get("patch_0")
    torch.testing.assert_close(m, mean)
    torch.testing.assert_close(s, std)
    assert len(store) == 1
    store.clear()
    assert len(store) == 0
