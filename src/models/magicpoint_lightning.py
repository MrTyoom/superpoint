import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.nn import BCELoss
from torchmetrics import MeanMetric

from src.loss_utils.loss import detector_loss_calculation
from src.metrics import metric_calculation
from src.models.superpoint import SuperPoint
from src.types import Tensor


class MagicPointLightning(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self._descriptor = False
        self._criterion = BCELoss(reduction="none")
        self._net = SuperPoint()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_precision = MeanMetric()
        self.val_recall = MeanMetric()

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def forward(self, synth_data: Tensor) -> Tensor:
        return self._net(batch=synth_data, descriptor=self._descriptor)

    def training_step(self, sample: tuple[Tensor, ...]) -> Tensor:
        img, masks, labels = sample
        semi = self(img)

        loss = detector_loss_calculation(semi, labels, masks, self._criterion, self.hparams.cell_size)
        self.train_loss(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log("train/loss", loss, on_step=False, on_epoch=True)

    def validation_step(self, sample: tuple[Tensor, ...]) -> None:
        img, masks, labels = sample
        semi = self(img)

        loss = detector_loss_calculation(semi, labels, masks, self._criterion, self.hparams.cell_size)
        self.val_loss(loss)

        precision, recall = metric_calculation(
            semi,
            labels,
            masks,
            self.hparams.detection_threshold,
            self.hparams.nms_dist,
        )

        self.val_precision(precision)
        self.val_recall(recall)

    def on_validation_epoch_end(self) -> None:
        avg_loss = self.val_loss.compute()
        avg_precision = self.val_precision.compute()
        avg_recall = self.val_recall.compute()

        self.log("val/loss", avg_loss, on_step=False, on_epoch=True)
        self.log("val/precision", avg_precision, on_step=False, on_epoch=True)
        self.log("val/recall", avg_recall, on_step=False, on_epoch=True)

        self.val_loss.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.hparams.learning_rate)
        return optimizer
