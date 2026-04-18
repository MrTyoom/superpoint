import rootutils
import torch
from dvclive.lightning import DVCLiveLogger
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from omegaconf import DictConfig, OmegaConf


rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.models.magicpoint_lightning import MagicPointLightning
from src.synthetic_loader import Loader, SyntheticDataset


REFRESH_RATE = 50

torch.set_float32_matmul_precision("high")


def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    loader = Loader(cfg, SyntheticDataset)

    model = MagicPointLightning(cfg)

    callbacks = [
        ModelCheckpoint(dirpath=cfg.log_dir, save_top_k=2, monitor="val/precision", mode="max", save_last=True),
        RichProgressBar(refresh_rate=REFRESH_RATE),
    ]

    logger = DVCLiveLogger(prefix="magicpoint", log_model=False, dir=cfg.log_dir)

    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        limit_val_batches=cfg.limit_val_batches,
        default_root_dir=cfg.log_dir,
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    trainer.fit(model, loader)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.train_magicpoint

    main(cfg)
