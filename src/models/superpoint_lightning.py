import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanMetric

from src.loss_utils.loss import SuperPointLoss
from src.metrics import metric_calculation
from src.models.superpoint import SuperPoint
from src.transform import Augmentation
from src.types import Tensor


def load_magic_point_weights(weights_path: str) -> SuperPoint:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.removeprefix("_net."): v for k, v in checkpoint["state_dict"].items() if k.startswith("_net.")}
    model = SuperPoint()
    model.load_state_dict(weights, strict=True)
    return model


class SuperPointLightning(LightningModule):  # noqa: WPS230
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self._descriptor = True
        self._criterion = SuperPointLoss()
        self._net = SuperPoint()

        self.train_detector_loss = MeanMetric()
        self.train_descriptor_loss = MeanMetric()
        self.train_loss = MeanMetric()

        self.val_detector_loss = MeanMetric()
        self.val_descriptor_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_precision = MeanMetric()
        self.val_recall = MeanMetric()

        self.augmentation = Augmentation(cfg.augmentation)

        # загрузка предобученных весов MagicPoint
        if cfg.get("magic_point_model"):
            self._net = load_magic_point_weights(cfg.magic_point_model)

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def forward(self, sat_data: Tensor) -> Tensor:
        return self._net(batch=sat_data, descriptor=self._descriptor)

    def training_step(self, sample: tuple[Tensor, Tensor]) -> Tensor:
        images, labels = sample
        src_batch = images.float().unsqueeze(1) / 255.0
        src_labels = labels.float().unsqueeze(1)

        # TODO: проверить вычисление гомографий и аугментацию
        dst_batch, dst_labels, homographs = self.augmentation.augment_and_crop(src_batch, src_labels, geometry_aug=True)
        src_batch, src_labels, _ = self.augmentation.augment_and_crop(src_batch, src_labels, geometry_aug=False)

        # TODO: удалить этот код после проверки аугментации и гомографий
        # debug = True
        # if debug:
        #     import cv2
        #     import numpy as np
        #     from src.common import make_dir
        #     out_dir = make_dir("./tmp", delete_if_exist=True)
        #     _src_batch = (255 * src_batch[:, 0].cpu().numpy()).astype(np.uint8)
        #     _dst_batch = (255 * dst_batch[:, 0].cpu().numpy()).astype(np.uint8)
        #     _src_labels = (255 * src_labels[:, 0].cpu().numpy()).astype(np.uint8)
        #     _dst_labels = (255 * dst_labels[:, 0].cpu().numpy()).astype(np.uint8)
        #     for i in range(src_batch.shape[0]):
        #         s = str(i).zfill(2)
        #         cv2.imwrite(str(out_dir / f"{s}_src.jpg"), _src_batch[i])
        #         cv2.imwrite(str(out_dir / f"{s}_dst.jpg"), _dst_batch[i])
        #         cv2.imwrite(str(out_dir / f"{s}_src.png"), _src_labels[i])
        #         cv2.imwrite(str(out_dir / f"{s}_dst.png"), _dst_labels[i])
        #     ...

        src_semi, src_desc = self(src_batch)
        dst_semi, dst_desc = self(dst_batch)

        detector_loss, descriptor_loss = self._criterion(
            src_semi, dst_semi, src_desc, dst_desc, src_labels, dst_labels, homographs
        )

        loss = detector_loss + descriptor_loss

        self.train_descriptor_loss(descriptor_loss)
        self.train_detector_loss(detector_loss)
        self.train_loss(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute()
        detector_loss = self.train_detector_loss.compute()
        descriptor_loss = self.train_descriptor_loss.compute()

        self.train_loss.reset()
        self.train_detector_loss.reset()
        self.train_descriptor_loss.reset()

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/detector_loss", detector_loss, on_step=False, on_epoch=True)
        self.log("train/descriptor_loss", descriptor_loss, on_step=False, on_epoch=True)

    def validation_step(self, sample: dict):
        images, labels = sample

        src_batch = images.float().unsqueeze(1) / 255.0
        src_labels = labels.float().unsqueeze(1)

        dst_batch, dst_labels, homographs = self.augmentation.augment_and_crop(src_batch, src_labels, geometry_aug=True)

        src_batch, src_labels, _ = self.augmentation.augment_and_crop(src_batch, src_labels, geometry_aug=False)

        src_semi, src_desc = self(src_batch)
        dst_semi, dst_desc = self(dst_batch)

        detector_loss, descriptor_loss = self._criterion(
            src_semi, dst_semi, src_desc, dst_desc, src_labels, dst_labels, homographs
        )

        loss = detector_loss + descriptor_loss

        self.val_loss(loss)
        self.val_detector_loss(detector_loss)
        self.val_descriptor_loss(descriptor_loss)

        precision, recall = metric_calculation(
            src_semi,
            src_labels,
            None,
            self.hparams.detection_threshold,
            self.hparams.nms_dist,
        )

        self.val_precision(precision)
        self.val_recall(recall)

    def on_validation_epoch_end(self) -> None:  # noqa: WPS213
        loss = self.val_loss.compute()
        detector_loss = self.val_detector_loss.compute()
        descriptor_loss = self.val_descriptor_loss.compute()

        self.val_loss.reset()
        self.val_detector_loss.reset()
        self.val_descriptor_loss.reset()

        precision = self.val_precision.compute()
        recall = self.val_recall.compute()

        self.val_precision.reset()
        self.val_recall.reset()

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/detector_loss", detector_loss, on_step=False, on_epoch=True)
        self.log("val/descriptor_loss", descriptor_loss, on_step=False, on_epoch=True)
        self.log("val/precision", precision, on_step=False, on_epoch=True)
        self.log("val/recall", recall, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.hparams.learning_rate)
        return optimizer
