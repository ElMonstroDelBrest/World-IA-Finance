"""Generate synthetic OHLCV data for testing Strate I."""

import torch
import numpy as np
from pathlib import Path


def generate_head_and_shoulders(
    T: int = 512, base_price: float = 100.0, amplitude: float = 10.0
) -> torch.Tensor:
    """Generate a Head & Shoulders pattern in OHLCV format. Returns (T, 5)."""
    t = np.arange(T)
    close = np.full(T, base_price, dtype=np.float32)

    # Left shoulder, head, right shoulder
    ls_start, ls_end = T // 10, T // 10 + T // 5
    h_start, h_end = ls_end + T // 20, ls_end + T // 20 + T // 3
    rs_start = h_end + T // 20
    rs_end = min(rs_start + T // 5, T - 1)

    ls_t = t[ls_start:ls_end]
    close[ls_start:ls_end] += amplitude * 0.6 * np.sin(
        np.pi * (ls_t - ls_start) / (ls_end - ls_start)
    )

    h_t = t[h_start:h_end]
    close[h_start:h_end] += amplitude * np.sin(
        np.pi * (h_t - h_start) / (h_end - h_start)
    )

    rs_t = t[rs_start:rs_end]
    close[rs_start:rs_end] += amplitude * 0.6 * np.sin(
        np.pi * (rs_t - rs_start) / (rs_end - rs_start)
    )

    close = torch.from_numpy(close)
    close += torch.randn(T) * 0.01 * amplitude

    high = close + torch.rand(T) * 0.1 * amplitude
    low = close - torch.rand(T) * 0.1 * amplitude
    open_ = torch.cat([close[:1], close[:-1]])
    volume = torch.rand(T) * 1000 + 500

    return torch.stack([open_, high, low, close, volume], dim=1)


def generate_random_walk(
    T: int = 512, start_price: float = 100.0, volatility: float = 0.02
) -> torch.Tensor:
    """Generate geometric random walk OHLCV. Returns (T, 5)."""
    returns = torch.randn(T) * volatility
    price = start_price * torch.exp(torch.cumsum(returns, 0))

    high = price * (1 + torch.rand(T) * 0.01)
    low = price * (1 - torch.rand(T) * 0.01)
    open_ = torch.cat([price[:1], price[:-1]])
    volume = torch.rand(T) * 1000 + 500

    return torch.stack([open_, high, low, price, volume], dim=1)


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Generating synthetic OHLCV data...")

    for i in range(10):
        series = generate_random_walk(T=1024, start_price=np.random.uniform(50, 200))
        torch.save(series, data_dir / f"series_{i}.pt")

    for i in range(2):
        series = generate_head_and_shoulders(
            T=1024,
            base_price=np.random.uniform(80, 150),
            amplitude=np.random.uniform(10, 30),
        )
        torch.save(series, data_dir / f"series_{10 + i}.pt")

    print(f"Saved 12 series to {data_dir}/")
    print("  - 10 random walk, 2 head & shoulders")
    print("  - Each: (1024, 5) = (T, [O,H,L,C,V])")


if __name__ == "__main__":
    main()
