import rootutils
import torch
from dvclive.lightning import DVCLiveLogger
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.logger import LOG
from src.superpoint import MagicPointLightning
from src.synthetic_loader import Loader

REFRESH_RATE = 50

torch.set_float32_matmul_precision("medium")


def main(cfg: DictConfig) -> None:
    LOG.info("set seed: {0}".format(cfg.seed))
    seed_everything(cfg.seed)

    loader = Loader(cfg)
    model = MagicPointLightning(cfg)

    callbacks = [
        ModelCheckpoint(dirpath=cfg.log_dir, save_top_k=2, monitor="val/epoch_precision", mode="max", save_last=True),
        RichProgressBar(refresh_rate=REFRESH_RATE),
    ]

    logger = DVCLiveLogger(prefix="magicpoint", log_model=False)

    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        limit_val_batches=cfg.max_val_samples,
        default_root_dir=cfg.log_dir,
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    trainer.fit(model, loader)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["train_magicpoint"]

    main(cfg)
