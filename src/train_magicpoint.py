import rootutils
import torch
from dvclive.lightning import DVCLiveLogger
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from omegaconf import DictConfig, OmegaConf


rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.loaders.magicpoint_loader import MagicPointLoader
from src.models.magicpoint_lightning import MagicPointLightning


def main(cfg: DictConfig) -> None:
    # высокая точность умножения матриц для повышения производительности
    torch.set_float32_matmul_precision("high")

    # установка seed для воспроизводимости результатов
    seed_everything(cfg.seed)

    # загрузка данных и создание модели
    loader = MagicPointLoader(cfg)
    model = MagicPointLightning(cfg)

    # настройка коллбеков для сохранения модели и отображения прогресса
    callbacks = [
        ModelCheckpoint(dirpath=cfg.log_dir, save_top_k=2, monitor="val/precision", mode="max", save_last=True),
        RichProgressBar(cfg.refresh_rate),
    ]

    # настройка логгера для отслеживания метрик и сохранения результатов
    logger = DVCLiveLogger(prefix="magicpoint", log_model=False, dir=cfg.log_dir)

    # настройка и запуск обучения модели
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
