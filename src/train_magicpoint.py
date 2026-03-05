import rootutils
from dvclive.lightning import DVCLiveLogger
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.logger import LOG
from src.superpoint import MagicPointLightning
from src.synthetic_loader import Loader

REFRESH_RATE = 50


def main(cfg: DictConfig) -> None:
    LOG.info("set seed: {0}".format(cfg.seed))
    seed_everything(cfg.seed)

    loader = Loader(cfg)
    model = MagicPointLightning(cfg)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(save_top_k=2, monitor="val/epoch_precision", mode="max", every_n_epochs=1),
        RichProgressBar(refresh_rate=REFRESH_RATE),
    ]

    logger = DVCLiveLogger(
        dir="models", prefix="magicpoint", log_model="best", run_name=f"run_{cfg.seed}", save_dvc_exp=False
    )

    trainer = Trainer(
        max_steps=cfg.num_iters,
        default_root_dir="data/logs",
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(model, loader)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["train_magicpoint"]

    main(cfg)
