"""Lightning DataModule for Strate II."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .token_dataset import TokenSequenceDataset, SyntheticTokenDataset


class StrateIIDataModule(pl.LightningDataModule):
    """DataModule for Strate II pre-training.

    Uses pre-tokenized sequences from Strate I, or synthetic data for dev/test.

    Args:
        token_dir: Directory containing .pt token files.
        seq_len: Sequence length.
        num_codes: Codebook size (for synthetic).
        batch_size: Batch size.
        val_split: Fraction for validation.
        num_workers: DataLoader workers.
        synthetic: If True, use synthetic data instead of real tokens.
        num_synthetic: Number of synthetic sequences.
    """

    def __init__(
        self,
        token_dir: str = "data/tokens/",
        seq_len: int = 64,
        num_codes: int = 1024,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 4,
        synthetic: bool = False,
        num_synthetic: int = 512,
    ):
        super().__init__()
        self.token_dir = token_dir
        self.seq_len = seq_len
        self.num_codes = num_codes
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.synthetic = synthetic
        self.num_synthetic = num_synthetic

    def setup(self, stage: str | None = None):
        if self.synthetic:
            full = SyntheticTokenDataset(
                num_sequences=self.num_synthetic,
                seq_len=self.seq_len,
                num_codes=self.num_codes,
            )
        else:
            full = TokenSequenceDataset(
                token_dir=self.token_dir,
                seq_len=self.seq_len,
            )

        n_val = int(len(full) * self.val_split)
        n_train = len(full) - n_val
        self.train_ds, self.val_ds = random_split(
            full, [n_train, n_val],
            generator=__import__("torch").Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
