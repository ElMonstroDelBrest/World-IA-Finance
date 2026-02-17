import os
from glob import glob

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .transforms import PatchTransform


class OHLCVPatchDataset(Dataset):
    """Loads .pt OHLCV files, applies log-returns + patching, concatenates all patches."""

    def __init__(self, data_dir: str, patch_length: int = 16, stride: int = 16):
        super().__init__()
        self.transform = PatchTransform(patch_length, stride)
        self.patches = self._load(data_dir)

    def _load(self, data_dir: str) -> Tensor:
        file_paths = sorted(glob(os.path.join(data_dir, "*.pt")))
        if not file_paths:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

        all_patches = []
        for path in file_paths:
            raw = torch.load(path, weights_only=True)  # (T, 5)
            patches = self.transform(raw)  # (N, L, C)
            if patches.size(0) > 0:
                all_patches.append(patches)

        return torch.cat(all_patches, dim=0)

    def __len__(self) -> int:
        return self.patches.size(0)

    def __getitem__(self, idx: int) -> Tensor:
        return self.patches[idx]
