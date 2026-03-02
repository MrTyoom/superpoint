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

# isort: off
from train_utils.d2s import DepthToSpace, SpaceToDepth
from train_utils.train_utils import getPtsFromHeatmap, precisionRecall

# isort: on
rootutils.setup_root(__file__, indicator="src", pythonpath=True)


from src.logger import LOG
from src.synthetic_loader import Loader

EPS = 1e-8
REFRESH_RATE = 50
SEMI = "semi"


def get_heatmap(semi):
    semi_soft = nn.functional.softmax(semi, dim=1)
    nodust = semi_soft[:, :-1, :, :]

    depth_to_space = DepthToSpace(8)
    heatmap = depth_to_space(nodust)
    return heatmap


# def compute_precision_recall(
#     pred_points: list[Tensor], true_points: list[Tensor], dist_thresh: float
# ) -> tuple[float, float]:
#     total_true = sum(len(pts) for pts in true_points)
#     total_pred = sum(len(pts) for pts in pred_points)
#     total_correct = 0

#     for pred, true_el in zip(pred_points, true_points):
#         if len(pred) == 0 or len(true_el) == 0:
#             continue

#         for pred_el in pred:
#             dist = torch.norm(true_el - pred_el.unsqueeze(0), dim=1)
#             if (dist < dist_thresh).any():
#                 total_correct += 1

#     precision = total_correct / (total_pred + EPS)
#     recall = total_correct / (total_true + EPS)

#     return precision, recall


# def extract_points(heatmap: Tensor, threshold: float, nms_dist: int) -> list[Tensor]:
#     batch_pts = []
#     for batch in range(heatmap.shape[0]):
#         mask = heatmap[batch, 0] > threshold
#         coords = torch.nonzero(mask).float()

#         if len(coords) > 0:
#             boxes = torch.cat([coords - nms_dist // 2, coords + nms_dist // 2], dim=1)
#             scores = heatmap[batch, 0][mask]
#             keep = nms(boxes, scores, nms_dist)
#             coords = coords[keep]

#         batch_pts.append(coords)

#     return batch_pts


def get_labels(labels_two_dim: Tensor, cell_size: int) -> tuple[Tensor, Tensor]:
    """
    transform 2D labels to 3D shape for training
    """
    batch, _, img_h, img_w = labels_two_dim.shape
    Hc, Wc = img_h // cell_size, img_w // cell_size

    space2depth = SpaceToDepth(8)
    labels = space2depth(labels_two_dim)
    dustbin = torch.ones((batch, 1, Hc, Wc)).cuda()
    labels = torch.cat((labels * 2, dustbin.view(batch, 1, Hc, Wc)), dim=1)
    labels = torch.argmax(labels, dim=1)
    return labels


def get_masks(masks_two_dim: Tensor, cell_size: int) -> Tensor:
    """
    # 2D mask is constructed into 3D (Hc, Wc) space for training
    :param mask_2D:
        tensor [batch, 1, H, W]
    :param cell_size:
        8 (default)
    :return:
        flattened 3D mask for training
    """
    batch, _, img_h, img_w = masks_two_dim.shape
    Hc, Wc = img_h // cell_size, img_w // cell_size

    space2depth = SpaceToDepth(8)
    masks = space2depth(masks_two_dim)

    dustbin = masks.sum(dim=1)
    dustbin = 1 - dustbin
    dustbin[dustbin < 1.0 - EPS] = 0

    masks_three_dim = torch.cat((masks, dustbin.view(batch, 1, Hc, Wc)), dim=1)

    # normalize
    dn = masks_three_dim.sum(dim=1)
    masks_three_dim = masks_three_dim.div(torch.unsqueeze(dn, 1)).float()

    masks_three_dim_flat = torch.prod(masks, 1)
    return masks_three_dim_flat


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

        cPa = self._relu(self._bnPa(self._convPa(x4)))
        semi = self._bnPb(self._convPb(cPa))

        return {SEMI: semi}

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

        self._criterion = nn.CrossEntropyLoss(reduction="none")
        self._net = MagicPoint()
        self._max_validation_samples = cfg.max_val_samples

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.validation_samples: list[Any] = []

        self.test_precision = MeanMetric()
        self.test_recall = MeanMetric()

    def forward(self, synth_data: Tensor) -> Tensor:
        return self._net(synth_data)

    def training_step(self, sample: tuple[Tensor, ...]) -> Tensor:
        img, masks, labels = sample

        outs = self._net(img)
        semi = outs[SEMI]

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

            outs = self._net(img)
            semi = outs[SEMI]

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
        semi = outs[SEMI]

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


def create_dataloaders(cfg: DictConfig) -> tuple[Loader, Loader]:
    loader = Loader(cfg)
    loader.setup(stage="fit")
    train_loader = loader.train_dataloader()
    val_loader = loader.val_dataloader()

    LOG.info(f"train loader size: {len(train_loader)}")
    LOG.info(f"val loader size: {len(val_loader)}")

    return train_loader, val_loader


def main(cfg: DictConfig) -> None:
    LOG.info("set seed: {0}".format(cfg.seed))
    seed_everything(cfg.seed)

    loaders = create_dataloaders(cfg)

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

    trainer.fit(model, loaders[0], loaders[1])
    trainer.test(model, loaders[1])


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["train_magicpoint"]

    main(cfg)
