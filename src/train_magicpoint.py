from typing import Any

import rootutils
import torch
from dvclive.lightning import DVCLiveLogger
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from omegaconf import DictConfig, OmegaConf
from torch import nn  # noqa: WPS458
from torch.types import Tensor
from torchmetrics import MeanMetric
from torchvision.ops import nms

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.logger import LOG
from src.synthetic_loader import Loader

EPS = 1e-8


def get_heatmap(semi, image_size=None):
    semi_soft = nn.functional.softmax(semi, dim=1)

    heatmap = semi_soft[:, :-1, :, :]

    if image_size:
        heatmap = nn.functional.interpolate(heatmap, size=image_size, mode="bilinear", align_corners=True)

    return heatmap


def compute_precision_recall(
    pred_points: list[Tensor], true_points: list[Tensor], dist_thresh: float
) -> tuple[float, float]:
    total_true = sum(len(pts) for pts in true_points)
    total_pred = sum(len(pts) for pts in pred_points)
    total_correct = 0

    for pred, true_el in zip(pred_points, true_points):
        if len(pred) == 0 or len(true_el) == 0:
            continue

        for pred_el in pred:
            dist = torch.norm(true_el - pred_el.unsqueeze(0), dim=1)
            if (dist < dist_thresh).any():
                total_correct += 1

    precision = total_correct / (total_pred + EPS)
    recall = total_correct / (total_true + EPS)

    return precision, recall


def extract_points(heatmap: Tensor, threshold: float, nms_dist: int) -> list[Tensor]:
    batch_pts = []
    for batch in range(heatmap.shape[0]):
        mask = heatmap[batch, 0] > threshold
        coords = torch.nonzero(mask).float()

        if len(coords) > 0:
            boxes = torch.cat([coords - nms_dist // 2, coords + nms_dist // 2], dim=1)
            scores = heatmap[batch, 0][mask]
            keep = nms(boxes, scores, nms_dist)
            coords = coords[keep]

        batch_pts.append(coords)

    return batch_pts


def labels_to_classes(labels_two_dim: Tensor, masks_two_dim: Tensor, cell_size: int) -> tuple[Tensor, Tensor]:
    batch, _, img_h, img_w = labels_two_dim.shape
    Hc, Wc = img_h // cell_size, img_w // cell_size

    mask_pooled = nn.functional.avg_pool2d(masks_two_dim.float(), cell_size, cell_size)
    cell_mask = (mask_pooled.squeeze(1) > 0.5 - EPS).float()
    valid_labels = labels_two_dim * masks_two_dim

    labels = valid_labels.view(batch, 1, Hc, cell_size, Wc, cell_size)
    labels = labels.permute(0, 1, 3, 5, 2, 4).contiguous()
    labels = labels.view(batch, cell_size * cell_size, Hc, Wc)

    has_point = (labels > 0).float()
    position_indices = torch.arange(cell_size * cell_size, device=labels.device)
    position_indices = position_indices.view(1, -1, 1, 1)
    position_indices = position_indices.expand_as(labels)

    masked_positions = position_indices * has_point
    first_point = masked_positions.max(dim=1)[0].long()
    any_point = (has_point.sum(dim=1) > 0).float()

    labels_classes = torch.where(
        any_point > 0, first_point, torch.ones((batch, Hc, Wc), device=labels.device, dtype=torch.long) * 64
    )

    labels_classes = torch.where(cell_mask > 0, labels_classes, torch.ones_like(labels_classes) * 64)

    return labels_classes, cell_mask


class MagicPoint(nn.Module):
    def __init__(self):
        super().__init__()

        channels = [64, 64, 128, 128, 256]
        det_h = 65

        self.inc = self._double_conv(1, channels[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), self._double_conv(channels[0], channels[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), self._double_conv(channels[1], channels[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), self._double_conv(channels[2], channels[3]))

        self._relu = nn.ReLU(inplace=True)
        self._convPa = nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1)
        self._bnPa = nn.BatchNorm2d(channels[4])
        self._convPb = nn.Conv2d(channels[4], det_h, kernel_size=1, stride=1, padding=0)
        self._bnPb = nn.BatchNorm2d(det_h)

    def forward(self, input_vec):
        x1 = self.inc(input_vec)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))

        return semi

    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class MagicPointLightning(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self._criterion = nn.CrossEntropyLoss(reduce="none")
        self._net = MagicPoint()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_precision = MeanMetric()
        self.val_recall = MeanMetric()

    def forward(self, synth_data: Tensor) -> Tensor:
        return self._net(synth_data)

    def training_step(self, sample: tuple[Tensor, ...]) -> Tensor:
        img, masks, labels = sample

        outs = self._net(img)
        semi = outs["semi"]

        labels_classes, cell_mask = labels_to_classes(labels, masks, cfg.cell_size)
        loss_per_cell = self._criterion(semi, labels_classes)

        loss = (loss_per_cell * cell_mask).sum() / (cell_mask.sum() + EPS)

        self.train_loss(loss)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute()

        self.train_loss.reset()

        self.log("train/loss", loss)

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()

        self.log("val/epoch_loss", loss, on_epoch=True)
        self.log("val/epoch_precision", precision, on_epoch=True)
        self.log("val/epoch_recall", recall, on_epoch=True)

        self.val_loss.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def validation_step(self, sample: tuple[Tensor, ...]) -> None:
        img, masks, labels = sample
        outs = self(img)
        semi = outs["semi"]

        labels_classes, cell_mask = labels_to_classes(labels, masks, cfg.cell_size)

        loss_per_cell = self._criterion(semi, labels_classes)

        loss = (loss_per_cell * cell_mask).sum() / (cell_mask.sum() + EPS)
        heatmap = get_heatmap(semi, image_size=img.shape[2:])

        pred_points = extract_points(heatmap, threshold=self.cfg.detection_threshold, nms_dist=self.cfg.nms_dist)

        true_points = []
        for batch in range(labels.shape[0]):
            valid_labels = labels[batch, 0] * masks[batch, 0]
            points = torch.nonzero(valid_labels > 0).float()
            true_points.append(points)

        precision, recall = compute_precision_recall(pred_points, true_points, dist_thresh=self.cfg.nms_dist)

        self.val_loss(loss)
        self.val_precision(precision)
        self.val_recall(recall)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/precision", precision, on_epoch=True, prog_bar=True)
        self.log("val/recall", recall, on_epoch=True, prog_bar=True)

    def test_step(self, sample: tuple[Tensor, ...]) -> None:
        img, masks, labels = sample

        outs = self(img)
        semi = outs["semi"]

        heatmap = get_heatmap(semi, image_size=img.shape[2:])

        pred_points = extract_points(heatmap, threshold=self.cfg.detection_threshold, nms_dist=self.cfg.nms_dist)

        true_points = []
        for batch in range(labels.shape[0]):
            valid_labels = labels[batch, 0] * masks[batch, 0]
            points = torch.nonzero(valid_labels > 0).float()
            true_points.append(points)

        precision, recall = compute_precision_recall(pred_points, true_points, dist_thresh=self.cfg.nms_dist)

        self.test_precision(precision)
        self.test_recall(recall)

        self.log("test/precision", precision, on_step=True)
        self.log("test/recall", recall, on_step=True)

    def on_test_epoch_end(self) -> None:
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()

        self.log("test/epoch_precision", precision)
        self.log("test/epoch_recall", recall)

        self.test_precision.reset()
        self.test_recall.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate, betas=(0.9, 0.999))

        return optimizer


def main(cfg: DictConfig) -> None:
    LOG.info("set seed: {0}".format(cfg.seed))
    seed_everything(cfg.seed)

    loader = Loader(cfg)
    loader.setup(stage="fit")

    train_loader = loader.train_dataloader()
    LOG.info("train loader size: {0}".format(len(train_loader)))

    val_loader = loader.val_dataloader()
    LOG.info("val loader size: {0}".format(len(val_loader)))

    model = MagicPointLightning(cfg)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(save_top_k=2, monitor="val/metric", mode="max", every_n_epochs=1),
        RichProgressBar(refresh_rate=50),
    ]

    logger = DVCLiveLogger(
        dir="data/logs", prefix="magicpoint", log_model="best", run_name=f"run_{cfg.seed}", save_dvc_exp=False
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
    trainer.test(model, loader)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["train_magicpoint"]

    main(cfg)
