import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanMetric

from src.loss_utils.loss import SuperPointLoss
from src.metrics import metric_calculation
from src.models.superpoint import SuperPoint
from src.types import Tensor


def load_magic_point_weights(weights_path: str) -> SuperPoint:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.removeprefix("_net."): v for k, v in checkpoint["state_dict"].items() if k.startswith("_net.")}
    model = SuperPoint()
    model.load_state_dict(weights, strict=True)
    return model


class SuperPointLightning(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self._descriptor = True
        self._criterion = SuperPointLoss(cfg)
        self._net = SuperPoint()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_precision = MeanMetric()
        self.val_recall = MeanMetric()

        # загрузка предобученных весов MagicPoint
        if cfg.get("magic_point_model"):
            self._net = load_magic_point_weights(cfg.magic_point_model)

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def forward(self, sat_data: Tensor) -> Tensor:
        return self._net(batch=sat_data, descriptor=self._descriptor)

    def training_step(self, sample: dict) -> Tensor:
        img = sample["image"]
        img_w = sample["warped_img"]
        mask_two_dim = sample["mask"]
        mask_two_dim_w = sample["mask_w"]
        labels_two_dim = sample["labels"]
        labels_two_dim_w = sample["labels_w"]
        homo = sample["homo"]

        semi, desc = self(img)
        semi_w, desc_w = self(img_w)

        masks = (mask_two_dim, mask_two_dim_w)

        loss = self._criterion(semi, semi_w, desc, desc_w, labels_two_dim, labels_two_dim_w, masks, homo)

        self.train_loss(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log("train/loss", loss, on_step=False, on_epoch=True)

    def validation_step(self, sample: dict):
        img = sample["image"]
        img_w = sample["warped_img"]
        mask_two_dim = sample["mask"]
        mask_two_dim_w = sample["mask_w"]
        labels_two_dim = sample["labels"]
        labels_two_dim_w = sample["labels_w"]
        homo = sample["homo"]

        semi, desc = self(img)
        semi_w, desc_w = self(img_w)

        masks = (mask_two_dim, mask_two_dim_w)

        loss = self._criterion(semi, semi_w, desc, desc_w, labels_two_dim, labels_two_dim_w, masks, homo)

        self.val_loss(loss)

        precision, recall = metric_calculation(
            semi,
            labels_two_dim,
            mask_two_dim,
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
