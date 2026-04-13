import rootutils
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from omegaconf import OmegaConf


rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.logger import LOG
from src.models.superpoint import SuperPointLightning
from src.satellite_loader import SatelliteDataset
from src.synthetic_loader import Loader


REFRESH_RATE = 50

torch.set_float32_matmul_precision("high")


def main(cfg):
    LOG.info("set seed: {0}".format(cfg.seed))
    seed_everything(cfg.seed)

    loader = Loader(cfg, SatelliteDataset)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    model = SuperPointLightning(cfg_dict)

    callbacks = [
        ModelCheckpoint(dirpath=cfg.log_dir, save_top_k=2, monitor="val/precision", mode="max", save_last=True),
        RichProgressBar(refresh_rate=REFRESH_RATE),
    ]

    # logger = DVCLiveLogger(prefix="superpoint", log_model=False, dir=cfg.log_dir)

    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        limit_val_batches=cfg.limit_val_batches,
        num_sanity_val_steps=0,
        default_root_dir=cfg.log_dir,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    trainer.fit(model, loader)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.train_superpoint)
