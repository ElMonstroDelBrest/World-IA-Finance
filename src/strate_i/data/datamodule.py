import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from ..config import DataConfig, PatchConfig, TrainingConfig
from .dataset import OHLCVPatchDataset


class OHLCVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        patch_config: PatchConfig,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.data_config = data_config
        self.patch_config = patch_config
        self.training_config = training_config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None):
        dataset = OHLCVPatchDataset(
            data_dir=self.data_config.data_path,
            patch_length=self.patch_config.patch_length,
            stride=self.patch_config.stride,
        )
        total = len(dataset)
        val_len = int(self.data_config.val_split * total)
        train_len = total - val_len
        gen = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_len, val_len], generator=gen
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            persistent_workers=self.data_config.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            persistent_workers=self.data_config.num_workers > 0,
        )
