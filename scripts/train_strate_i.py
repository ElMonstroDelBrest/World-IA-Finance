import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.strate_i.config import load_config
from src.strate_i.data.datamodule import OHLCVDataModule
from src.strate_i.lightning_module import StrateILightningModule


def main():
    parser = argparse.ArgumentParser(description="Train Strate I Tokenizer")
    parser.add_argument("--config", type=str, default="configs/strate_i.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    datamodule = OHLCVDataModule(
        data_config=config.data,
        patch_config=config.patch,
        training_config=config.training,
    )
    model = StrateILightningModule(config)

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss/total",
            dirpath="checkpoints/",
            filename="strate-i-{epoch:02d}-{val/loss/total:.4f}",
            save_top_k=3,
            mode="min",
        ),
        EarlyStopping(
            monitor="val/loss/total",
            patience=config.training.patience,
            mode="min",
        ),
    ]
    logger = TensorBoardLogger("tb_logs", name="strate_i")

    trainer = pl.Trainer(
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
