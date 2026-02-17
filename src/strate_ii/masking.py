"""Block masking for Fin-JEPA.

Generates contiguous block masks for JEPA-style self-supervised learning.
Masked positions are target tokens; unmasked are context tokens.
"""

import torch
from torch import Tensor


def generate_block_mask(
    seq_len: int,
    mask_ratio: float = 0.5,
    block_size_min: int = 4,
    block_size_max: int = 8,
    device: torch.device | None = None,
) -> Tensor:
    """Generate a single block mask for one sequence.

    Greedily places random-sized blocks until target mask ratio is reached.

    Args:
        seq_len: Length of the sequence.
        mask_ratio: Target fraction of positions to mask.
        block_size_min: Minimum block size.
        block_size_max: Maximum block size.
        device: Target device.

    Returns:
        Boolean tensor (S,) where True = masked (target).
    """
    target_masked = int(seq_len * mask_ratio)
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    masked_count = 0

    # Collect available positions
    available = list(range(seq_len))

    while masked_count < target_masked and available:
        block_size = torch.randint(block_size_min, block_size_max + 1, (1,)).item()
        # Pick a random start from remaining available positions
        start_idx = torch.randint(0, len(available), (1,)).item()
        start = available[start_idx]
        end = min(start + block_size, seq_len)

        mask[start:end] = True
        masked_count = mask.sum().item()

        # Remove newly masked positions from available
        available = [i for i in available if not mask[i]]

    return mask


def generate_batch_masks(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.5,
    block_size_min: int = 4,
    block_size_max: int = 8,
    device: torch.device | None = None,
) -> Tensor:
    """Generate block masks for a batch.

    Args:
        batch_size: Number of sequences.
        seq_len: Length of each sequence.
        mask_ratio: Target fraction of positions to mask.
        block_size_min: Minimum block size.
        block_size_max: Maximum block size.
        device: Target device.

    Returns:
        Boolean tensor (B, S) where True = masked (target).
    """
    masks = torch.stack([
        generate_block_mask(seq_len, mask_ratio, block_size_min, block_size_max, device)
        for _ in range(batch_size)
    ])
    return masks
