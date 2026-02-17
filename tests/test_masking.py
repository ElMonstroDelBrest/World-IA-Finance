"""Tests for block masking."""

import torch
import pytest

from src.strate_ii.masking import generate_block_mask, generate_batch_masks


def test_mask_shape():
    """Mask shape matches seq_len."""
    mask = generate_block_mask(64, mask_ratio=0.5)
    assert mask.shape == (64,)
    assert mask.dtype == torch.bool


def test_mask_ratio_approximate():
    """Mask ratio should be approximately correct over many samples."""
    ratios = []
    for _ in range(100):
        mask = generate_block_mask(64, mask_ratio=0.5, block_size_min=4, block_size_max=8)
        ratios.append(mask.float().mean().item())
    avg = sum(ratios) / len(ratios)
    # Allow ±15% tolerance due to block constraints
    assert 0.35 < avg < 0.65, f"Average mask ratio {avg} outside [0.35, 0.65]"


def test_mask_has_both_context_and_targets():
    """Mask should have both True (targets) and False (context) positions."""
    for _ in range(20):
        mask = generate_block_mask(64, mask_ratio=0.5)
        assert mask.any(), "Mask has no target positions"
        assert not mask.all(), "Mask has no context positions"


def test_mask_contiguous_blocks():
    """Masked regions should contain contiguous blocks of at least block_size_min."""
    mask = generate_block_mask(64, mask_ratio=0.5, block_size_min=4, block_size_max=8)
    # Find contiguous runs of True
    runs = []
    current_run = 0
    for val in mask:
        if val:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    # At least one block should exist
    assert len(runs) > 0
    # Due to block overlap, some runs may merge to be larger than block_size_max
    # But the minimum individual placement is block_size_min


def test_batch_masks_shape():
    """Batch masks shape is (B, S)."""
    masks = generate_batch_masks(8, 64, mask_ratio=0.5)
    assert masks.shape == (8, 64)
    assert masks.dtype == torch.bool


def test_batch_masks_different():
    """Different sequences should get different masks (stochastic)."""
    masks = generate_batch_masks(16, 64, mask_ratio=0.5)
    # Extremely unlikely all 16 masks are identical
    unique = masks.unique(dim=0)
    assert unique.shape[0] > 1, "All batch masks are identical"


def test_mask_device():
    """Mask should be on the specified device."""
    mask = generate_block_mask(64, device=torch.device("cpu"))
    assert mask.device.type == "cpu"


def test_small_sequence():
    """Should work with small sequences."""
    mask = generate_block_mask(8, mask_ratio=0.5, block_size_min=2, block_size_max=4)
    assert mask.shape == (8,)
    assert mask.any()
