import pytest
import torch

from src.strate_i.losses import VQVAELoss, soft_dtw_loss


@pytest.fixture
def loss_fn():
    return VQVAELoss(huber_delta=1.0, sdtw_alpha=0.1, sdtw_gamma=0.1)


def test_huber_component(loss_fn):
    x = torch.randn(8, 16, 5)
    x_hat = x + torch.randn_like(x) * 0.1
    commitment = torch.tensor(0.1)
    out = loss_fn(x, x_hat, commitment)
    assert out["huber"].item() > 0


def test_sdtw_component():
    x = torch.randn(4, 16, 5)
    x_hat = x + torch.randn_like(x) * 0.1
    loss = soft_dtw_loss(x, x_hat, gamma=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_total_loss_components(loss_fn):
    x = torch.randn(4, 16, 5)
    x_hat = x + torch.randn_like(x) * 0.1
    commitment = torch.tensor(0.1)
    out = loss_fn(x, x_hat, commitment)
    expected = out["huber"] + out["sdtw"] + out["commitment"]
    torch.testing.assert_close(out["total"], expected)


def test_zero_huber_on_identical(loss_fn):
    x = torch.randn(4, 16, 5)
    commitment = torch.tensor(0.0)
    out = loss_fn(x, x.clone(), commitment)
    assert out["huber"].item() < 1e-6
