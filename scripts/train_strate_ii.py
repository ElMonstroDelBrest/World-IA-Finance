"""Train Strate II (Fin-JEPA) model.

Usage:
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml

    # Synthetic data for development:
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml --synthetic

    # With Strate I codebook:
    python scripts/train_strate_ii.py --config configs/strate_ii.yaml \
        --codebook_checkpoint checkpoints/strate_i_best.ckpt
"""

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.strate_ii.config import load_config
from src.strate_ii.data.datamodule import StrateIIDataModule
from src.strate_ii.lightning_module import StrateIILightningModule


def main():
    parser = argparse.ArgumentParser(description="Train Strate II (Fin-JEPA)")
    parser.add_argument("--config", type=str, default="configs/strate_ii.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--num_synthetic", type=int, default=512)
    parser.add_argument(
        "--codebook_checkpoint", type=str, default=None,
        help="Path to Strate I checkpoint (to load codebook weights)",
    )
    parser.add_argument(
        "--strate_i_config", type=str, default="configs/strate_i_binance.yaml",
        help="Path to Strate I config (for codebook loading)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Data
    datamodule = StrateIIDataModule(
        token_dir=config.data.token_dir,
        seq_len=config.embedding.seq_len,
        num_codes=config.embedding.num_codes,
        batch_size=config.training.batch_size,
        val_split=config.data.val_split,
        num_workers=config.data.num_workers,
        synthetic=args.synthetic,
        num_synthetic=args.num_synthetic,
    )

    # Model
    model = StrateIILightningModule(config)

    # Load codebook from Strate I if provided
    if args.codebook_checkpoint:
        from src.strate_i.config import load_config as load_strate_i_config
        from src.strate_i.lightning_module import StrateILightningModule

        strate_i_config = load_strate_i_config(args.strate_i_config)
        strate_i = StrateILightningModule.load_from_checkpoint(
            args.codebook_checkpoint, config=strate_i_config
        )
        codebook_weights = strate_i.tokenizer.vqvae.codebook.embeddings.clone()
        model.jepa.load_codebook(codebook_weights)
        print(f"Loaded codebook from {args.codebook_checkpoint}")

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/strate_ii/",
        filename="strate_ii-{epoch:03d}-{val/loss/total:.4f}",
        monitor="val/loss/total",
        mode="min",
        save_top_k=3,
    )

    # Logger
    logger = TensorBoardLogger("tb_logs", name="strate_ii")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        precision=config.training.precision,
        callbacks=[checkpoint_cb],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
