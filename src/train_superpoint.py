import rootutils
import torch
from dvclive.lightning import DVCLiveLogger
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from omegaconf import OmegaConf


rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.loaders.superpoint_loader import SuperPointLoader
from src.models.superpoint_lightning import SuperPointLightning


def main(cfg):
    # установка seed для воспроизводимости результатов
    seed_everything(cfg.seed)

    # высокая точность умножения матриц для повышения производительности
    torch.set_float32_matmul_precision("high")

    # загрузка данных и создание модели
    loader = SuperPointLoader(cfg)
    model = SuperPointLightning(cfg)

    # настройка коллбеков для сохранения модели и отображения прогресса
    callbacks = [
        ModelCheckpoint(dirpath=cfg.log_dir, save_top_k=2, monitor="val/precision", mode="max", save_last=True),
        RichProgressBar(cfg.refresh_rate),
    ]

    # настройка логгера для отслеживания метрик и сохранения результатов
    logger = DVCLiveLogger(prefix="superpoint", log_model=False, dir=cfg.log_dir)

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
    main(cfg.train_superpoint)
