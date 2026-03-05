from typing import Any

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.nn import BatchNorm2d, Conv2d, CrossEntropyLoss, MaxPool2d, Module, ReLU
from torch.types import Tensor
from torchmetrics import MeanMetric

from src.logger import LOG
from src.loss import get_labels, get_masks
from src.metrics import get_heatmap
from src.train_utils.train_utils import getPtsFromHeatmap, precisionRecall

EPS = 1e-8
SEMI = "semi"


class SuperPoint(Module):
    def __init__(self):
        super().__init__()

        c1 = 64
        c2 = 64
        c3 = 128
        c4 = 128
        c5 = 256
        d1 = 256
        det_h = 65

        self.relu = ReLU(inplace=True)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        self._bn1a = BatchNorm2d(c1)
        self._bn1b = BatchNorm2d(c1)
        self._bn2a = BatchNorm2d(c2)
        self._bn2b = BatchNorm2d(c2)
        self._bn3a = BatchNorm2d(c3)
        self._bn3b = BatchNorm2d(c3)
        self._bn4a = BatchNorm2d(c4)
        self._bn4b = BatchNorm2d(c4)
        self._bnPa = BatchNorm2d(c5)
        self._bnPb = BatchNorm2d(det_h)
        self._bnDa = BatchNorm2d(c5)
        self._bnDb = BatchNorm2d(d1)

        # Encoder.
        self._conv1a = Conv2d(1, c1, 3, padding=1)
        self._conv1b = Conv2d(c1, c1, 3, padding=1)
        self._conv2a = Conv2d(c1, c2, 3, padding=1)
        self._conv2b = Conv2d(c2, c2, 3, padding=1)
        self._conv3a = Conv2d(c2, c3, 3, padding=1)
        self._conv3b = Conv2d(c3, c3, 3, padding=1)
        self._conv4a = Conv2d(c3, c4, 3, padding=1)
        self._conv4b = Conv2d(c4, c4, 3, padding=1)

        # Detector Head.
        self.convPa = Conv2d(c4, c5, kernel_size=3, padding=1)
        self.convPb = Conv2d(c5, det_h, kernel_size=1, padding=0)

        # Descriptor Head.
        self.convDa = Conv2d(c4, c5, kernel_size=3, padding=1)
        self.convDb = Conv2d(c5, d1, kernel_size=1, padding=0)

    def forward(self, batch: Tensor, descriptor=True):
        # Encoder
        x1 = self.relu(self._bn1a(self._conv1a(batch)))
        x2 = self.relu(self._bn1b(self._conv1b(x1)))
        x3 = self.pool(x2)

        x4 = self.relu(self._bn2a(self._conv2a(x3)))
        x5 = self.relu(self._bn2b(self._conv2b(x4)))
        x6 = self.pool(x5)

        x7 = self.relu(self._bn3a(self._conv3a(x6)))
        x8 = self.relu(self._bn3b(self._conv3b(x7)))
        x9 = self.pool(x8)

        x10 = self.relu(self._bn4a(self._conv4a(x9)))
        feats = self.relu(self._bn4b(self._conv4b(x10)))

        # Detector Head
        cPa = self.relu(self._bnPa(self.convPa(feats)))
        semi = self._bnPb(self.convPb(cPa))

        if not descriptor:
            return semi

        # Descriptor Head
        cDa = self.relu(self._bnDa(self.convDa(feats)))
        desc = self._bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        return semi, desc


class MagicPointLightning(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self._descriptor = False
        self._criterion = CrossEntropyLoss(reduction="none")
        self._net = SuperPoint()
        self._max_validation_samples = cfg.max_val_samples

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.validation_samples: list[Any] = []

        self.test_precision = MeanMetric()
        self.test_recall = MeanMetric()

    def forward(self, synth_data: Tensor) -> Tensor:
        return self._net(batch=synth_data, descriptor=self._descriptor)

    def training_step(self, sample: tuple[Tensor, ...]) -> Tensor:
        img, masks, labels = sample

        outs = self(img)
        semi = outs

        labels_three_dim = get_labels(labels_two_dim=labels, cell_size=self.cfg.cell_size)
        mask_three_dim_flat = get_masks(masks_two_dim=masks, cell_size=self.cfg.cell_size)

        loss_per_cell = self._criterion(semi, labels_three_dim)
        loss = (loss_per_cell * mask_three_dim_flat).sum() / (mask_three_dim_flat.sum() + EPS)

        self.train_loss(loss)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log("train/loss_epoch", loss)

    def validation_step(self, sample: tuple[Tensor, ...]) -> None:
        with torch.no_grad():
            img, masks, labels = sample

            outs = self(img)
            semi = outs

            labels_three_dim = get_labels(labels_two_dim=labels, cell_size=self.cfg.cell_size)
            mask_three_dim_flat = get_masks(masks_two_dim=masks, cell_size=self.cfg.cell_size)

            loss_per_cell = self._criterion(semi, labels_three_dim)
            loss = (loss_per_cell * mask_three_dim_flat).sum() / (mask_three_dim_flat.sum() + EPS)

            self.val_loss(loss)
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

            if len(self.validation_samples) < self._max_validation_samples:
                self.validation_samples.append(
                    {
                        "semi": semi.detach().cpu(),
                        "img": img.detach().cpu(),
                        "labels": labels.detach().cpu(),
                        "masks": masks.detach().cpu(),
                    }
                )
                LOG.info(f"Saved validation sample {len(self.validation_samples)}/{self._max_validation_samples}")

    def on_validation_epoch_end(self) -> None:
        avg_loss = self.val_loss.compute()
        self.val_loss.reset()
        self.log("val/epoch_loss", avg_loss)

        if self.validation_samples:
            self._compute_validation_metrics(avg_loss)

        self.validation_samples.clear()

    def test_step(self, sample: tuple[Tensor, ...]) -> None:
        img, masks, labels = sample

        outs = self(img)
        semi = outs

        heatmap = get_heatmap(semi)

        for batch_idx in range(semi.shape[0]):
            heatmap_np = heatmap[batch_idx, 0]
            heatmap_np = heatmap_np.cpu().numpy()
            pts_nms = getPtsFromHeatmap(heatmap_np, self.cfg.detection_threshold, self.cfg.nms_dist)

            pred_map = torch.zeros_like(labels[batch_idx, 0])
            if pts_nms.shape[1] > 0:
                pred_map[pts_nms[1, :].long(), pts_nms[0, :].long()] = 1

            gt_map = (labels[batch_idx, 0] * masks[batch_idx, 0]).cpu()

            pr = precisionRecall(pred_map, gt_map)

            self.test_precision(pr["precision"])
            self.test_recall(pr["recall"])

            self.log("test/precision", pr["precision"], on_step=True)
            self.log("test/recall", pr["recall"], on_step=True)

    def on_test_epoch_end(self) -> None:
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()

        self.log("test/epoch_precision", precision)
        self.log("test/epoch_recall", recall)

        self.test_precision.reset()
        self.test_recall.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.cfg.learning_rate, betas=(0.9, 0.999))
        return optimizer

    def _compute_validation_metrics(self, avg_loss: MeanMetric) -> None:
        all_precision = []
        all_recall = []

        LOG.info(f"Computing precision/recall on {len(self.validation_samples)} validation batches...")

        for batch in self.validation_samples:
            semi = batch["semi"].to(self.device)
            img = batch["img"].to(self.device)
            labels = batch["labels"].to(self.device)
            masks = batch["masks"].to(self.device)

            heatmap = get_heatmap(semi)

            for batch in range(semi.shape[0]):
                heatmap_np = heatmap[batch, 0]
                heatmap_np = heatmap_np.cpu().numpy()
                pts_nms = getPtsFromHeatmap(heatmap_np, self.cfg.detection_threshold, self.cfg.nms_dist)  # (2, N)

                pred_map = torch.zeros_like(labels[batch, 0]).cpu()
                pts_nms = torch.from_numpy(pts_nms).long()
                if pts_nms.shape[1] > 0:
                    pred_map[pts_nms[1, :].long(), pts_nms[0, :].long()] = 1

                gt_map = (labels[batch, 0] * masks[batch, 0]).cpu()

                pr = precisionRecall(pred_map, gt_map)
                all_precision.append(pr["precision"])
                all_recall.append(pr["recall"])

            del semi, img, labels, masks, heatmap

        if all_precision:
            avg_precision = sum(all_precision) / len(all_precision)
            avg_recall = sum(all_recall) / len(all_recall)

            self.log("val/epoch_precision", avg_precision)
            self.log("val/epoch_recall", avg_recall)

            LOG.info(f"Validation - Loss: {avg_loss:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
