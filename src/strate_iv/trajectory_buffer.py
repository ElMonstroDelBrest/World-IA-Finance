"""Trajectory buffer for Strate IV RL training.

Pre-computes episodes from MultiverseGenerator and stores them as .pt files.
The TrajectoryBuffer loads and samples entries for the LatentCryptoEnv.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class TrajectoryEntry:
    """A single pre-computed episode for RL training.

    All tensors are stored on CPU and moved to device as needed.
    """

    # Context tokens and masks
    context_tokens: Tensor       # (S,) int64
    weekend_mask: Tensor | None  # (S,) float or None

    # Raw OHLCV context for reference
    context_ohlcv: Tensor        # (T, 5) raw OHLCV

    # N future OHLCV trajectories from MultiverseGenerator
    future_ohlcv: Tensor         # (N, N_tgt, patch_len, 5)

    # N future latent representations
    future_latents: Tensor       # (N, N_tgt, d_model)

    # RevIN stats from context
    revin_means: Tensor          # (1, 5)
    revin_stds: Tensor           # (1, 5)

    # Last close price (anchor for returns)
    last_close: float

    # Context encoder pooled representation
    h_x_pooled: Tensor           # (d_model,)


class TrajectoryBuffer:
    """Loads and samples pre-computed TrajectoryEntry objects from disk."""

    def __init__(self, buffer_dir: str):
        self.buffer_dir = Path(buffer_dir)
        self.entries: list[TrajectoryEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.buffer_dir.exists():
            return
        pt_files = sorted(self.buffer_dir.glob("*.pt"))
        for f in pt_files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            entry = TrajectoryEntry(**data)
            self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def sample(self) -> TrajectoryEntry:
        """Sample a random trajectory entry."""
        return random.choice(self.entries)

    def add(self, entry: TrajectoryEntry) -> None:
        """Add an entry to the buffer (in-memory only)."""
        self.entries.append(entry)

    def split(
        self, val_ratio: float = 0.2, seed: int = 42
    ) -> tuple[TrajectoryBuffer, TrajectoryBuffer]:
        """Split buffer into train and eval buffers.

        Uses a deterministic shuffle so the split is reproducible.

        Args:
            val_ratio: Fraction of entries for eval (default 20%).
            seed: Random seed for the split.

        Returns:
            (train_buffer, eval_buffer)
        """
        rng = random.Random(seed)
        indices = list(range(len(self.entries)))
        rng.shuffle(indices)

        n_val = max(1, int(len(indices) * val_ratio))
        val_indices = set(indices[:n_val])

        train_buf = TrajectoryBuffer.__new__(TrajectoryBuffer)
        train_buf.entries = [e for i, e in enumerate(self.entries) if i not in val_indices]

        eval_buf = TrajectoryBuffer.__new__(TrajectoryBuffer)
        eval_buf.entries = [e for i, e in enumerate(self.entries) if i in val_indices]

        return train_buf, eval_buf


class TrajectoryPrecomputer:
    """Pre-computes trajectory entries using MultiverseGenerator + JEPA.

    Usage:
        precomputer = TrajectoryPrecomputer(generator, jepa, config)
        precomputer.run(token_dataset, ohlcv_dataset, output_dir)
    """

    def __init__(self, generator, jepa, n_futures: int = 16, n_tgt: int = 8):
        self.generator = generator
        self.jepa = jepa
        self.n_futures = n_futures
        self.n_tgt = n_tgt

    @torch.no_grad()
    def compute_entry(
        self,
        token_indices: Tensor,
        weekend_mask: Tensor | None,
        context_ohlcv: Tensor,
    ) -> TrajectoryEntry:
        """Compute a single trajectory entry.

        Args:
            token_indices: (S,) int64 context token indices.
            weekend_mask: (S,) float or None.
            context_ohlcv: (T, 5) raw OHLCV context.

        Returns:
            TrajectoryEntry with all fields populated.
        """
        device = token_indices.device

        # Add batch dimension
        tokens_b = token_indices.unsqueeze(0)          # (1, S)
        wm_b = weekend_mask.unsqueeze(0) if weekend_mask is not None else None
        ohlcv_b = context_ohlcv.unsqueeze(0)           # (1, T, 5)

        S = tokens_b.shape[1]
        target_positions = torch.arange(
            S - self.n_tgt, S, device=device
        ).unsqueeze(0)  # (1, N_tgt)

        # Generate futures via MultiverseGenerator
        result = self.generator.generate(
            tokens_b, wm_b, target_positions, ohlcv_b,
            n_samples=self.n_futures,
        )

        # Extract context encoder representation
        h_x = self.jepa.context_encoder(tokens_b, weekend_mask=wm_b)  # (1, S, d_model)
        h_x_pooled = h_x.mean(dim=1).squeeze(0)  # (d_model,)

        # RevIN stats
        means, stds = self.generator._estimate_context_stats(ohlcv_b)

        last_close = context_ohlcv[-1, 3].item()

        return TrajectoryEntry(
            context_tokens=token_indices.cpu(),
            weekend_mask=weekend_mask.cpu() if weekend_mask is not None else None,
            context_ohlcv=context_ohlcv.cpu(),
            future_ohlcv=result["ohlcv"][:, 0].cpu(),        # (N, N_tgt, patch_len, 5)
            future_latents=result["latents"][:, 0].cpu(),     # (N, N_tgt, d_model)
            revin_means=means[0].cpu(),                        # (1, 5)
            revin_stds=stds[0].cpu(),                          # (1, 5)
            last_close=last_close,
            h_x_pooled=h_x_pooled.cpu(),                       # (d_model,)
        )

    def run(
        self,
        dataset,
        ohlcv_lookup,
        output_dir: str,
        n_episodes: int = 255,
        device: str = "cpu",
    ) -> None:
        """Pre-compute and save trajectory entries.

        Args:
            dataset: TokenSequenceDataset yielding {token_indices, weekend_mask}.
            ohlcv_lookup: Callable(idx) -> (T, 5) raw OHLCV tensor.
            output_dir: Directory to save .pt files.
            n_episodes: Number of episodes to generate.
            device: Device to run on.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        indices = list(range(len(dataset)))
        if len(indices) > n_episodes:
            indices = random.sample(indices, n_episodes)
        elif len(indices) < n_episodes:
            indices = indices * (n_episodes // len(indices) + 1)
            indices = indices[:n_episodes]

        for i, idx in enumerate(indices):
            sample = dataset[idx]
            token_indices = sample["token_indices"].to(device)
            weekend_mask = sample.get("weekend_mask")
            if weekend_mask is not None:
                weekend_mask = weekend_mask.to(device)

            context_ohlcv = ohlcv_lookup(idx).to(device)

            entry = self.compute_entry(token_indices, weekend_mask, context_ohlcv)

            save_dict = {
                "context_tokens": entry.context_tokens,
                "weekend_mask": entry.weekend_mask,
                "context_ohlcv": entry.context_ohlcv,
                "future_ohlcv": entry.future_ohlcv,
                "future_latents": entry.future_latents,
                "revin_means": entry.revin_means,
                "revin_stds": entry.revin_stds,
                "last_close": entry.last_close,
                "h_x_pooled": entry.h_x_pooled,
            }
            torch.save(save_dict, out_path / f"episode_{i:05d}.pt")

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n_episodes}] episodes saved")
