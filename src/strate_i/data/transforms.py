import torch
from torch import Tensor


def compute_log_returns(ohlcv: Tensor) -> Tensor:
    """Log-returns for OHLC (indices 0-3), log(1 + vol/mean_vol) for volume (index 4).
    Input: (T, C=5), Output: (T-1, C=5).
    """
    if ohlcv.shape[0] < 2:
        return torch.empty(0, ohlcv.shape[1], device=ohlcv.device, dtype=ohlcv.dtype)

    prices = ohlcv[:, :4]
    price_log_returns = torch.log(prices[1:] + 1e-9) - torch.log(prices[:-1] + 1e-9)

    volume = ohlcv[:, 4]
    mean_volume = volume.mean()

    if mean_volume > 1e-8:
        volume_transform = torch.log1p(volume[1:] / mean_volume)
    else:
        volume_transform = torch.zeros_like(volume[1:])

    return torch.cat([price_log_returns, volume_transform.unsqueeze(-1)], dim=-1)


def extract_patches(series: Tensor, patch_length: int = 16, stride: int = 16) -> Tensor:
    """Extract patches from time series. Input: (T, C), Output: (N, L, C)."""
    if series.shape[0] < patch_length:
        return torch.empty(
            0, patch_length, series.shape[1], device=series.device, dtype=series.dtype
        )

    # unfold -> (N, C, L), then permute to (N, L, C)
    patches = series.unfold(0, patch_length, stride)
    return patches.permute(0, 2, 1)


class PatchTransform:
    def __init__(self, patch_length: int = 16, stride: int = 16):
        self.patch_length = patch_length
        self.stride = stride

    def __call__(self, ohlcv: Tensor) -> Tensor:
        log_returns = compute_log_returns(ohlcv)
        return extract_patches(log_returns, self.patch_length, self.stride)
