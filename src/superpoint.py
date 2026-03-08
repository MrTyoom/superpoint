import torch
from lightning import LightningModule
from torch.nn import BatchNorm2d, BCELoss, Conv2d, MaxPool2d, Module, ReLU
from torchmetrics import MeanMetric

from src.loss import loss_calculation
from src.metrics import batch_precision_recall, get_heatmap
from src.types import Tensor


class SuperPoint(Module):  # noqa: WPS230
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

        self.bn1a = BatchNorm2d(c1)
        self.bn1b = BatchNorm2d(c1)
        self.bn2a = BatchNorm2d(c2)
        self.bn2b = BatchNorm2d(c2)
        self.bn3a = BatchNorm2d(c3)
        self.bn3b = BatchNorm2d(c3)
        self.bn4a = BatchNorm2d(c4)
        self.bn4b = BatchNorm2d(c4)
        self.bnPa = BatchNorm2d(c5)
        self.bnPb = BatchNorm2d(det_h)
        self.bnDa = BatchNorm2d(c5)
        self.bnDb = BatchNorm2d(d1)

        # Encoder.
        self.conv1a = Conv2d(1, c1, 3, padding=1)
        self.conv1b = Conv2d(c1, c1, 3, padding=1)
        self.conv2a = Conv2d(c1, c2, 3, padding=1)
        self.conv2b = Conv2d(c2, c2, 3, padding=1)
        self.conv3a = Conv2d(c2, c3, 3, padding=1)
        self.conv3b = Conv2d(c3, c3, 3, padding=1)
        self.conv4a = Conv2d(c3, c4, 3, padding=1)
        self.conv4b = Conv2d(c4, c4, 3, padding=1)

        # Detector Head.
        self.convPa = Conv2d(c4, c5, kernel_size=3, padding=1)
        self.convPb = Conv2d(c5, det_h, kernel_size=1, padding=0)

        # Descriptor Head.
        self.convDa = Conv2d(c4, c5, kernel_size=3, padding=1)
        self.convDb = Conv2d(c5, d1, kernel_size=1, padding=0)

    def forward(self, batch: Tensor, descriptor=True):
        # Encoder
        x1 = self.relu(self.bn1a(self.conv1a(batch)))
        x2 = self.relu(self.bn1b(self.conv1b(x1)))
        x3 = self.pool(x2)

        x4 = self.relu(self.bn2a(self.conv2a(x3)))
        x5 = self.relu(self.bn2b(self.conv2b(x4)))
        x6 = self.pool(x5)

        x7 = self.relu(self.bn3a(self.conv3a(x6)))
        x8 = self.relu(self.bn3b(self.conv3b(x7)))
        x9 = self.pool(x8)

        x10 = self.relu(self.bn4a(self.conv4a(x9)))
        feats = self.relu(self.bn4b(self.conv4b(x10)))

        # Detector Head
        cPa = self.relu(self.bnPa(self.convPa(feats)))
        semi = self.bnPb(self.convPb(cPa))

        if not descriptor:
            return semi

        # Descriptor Head
        cDa = self.relu(self.bnDa(self.convDa(feats)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))

        return semi, desc


class MagicPointLightning(LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        self._descriptor = False
        self._criterion = BCELoss(reduction="none")
        self._net = SuperPoint()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_precision = MeanMetric()
        self.val_recall = MeanMetric()

        self.save_hyperparameters(cfg)

    def forward(self, synth_data: Tensor) -> Tensor:
        return self._net(batch=synth_data, descriptor=self._descriptor)

    def training_step(self, sample: tuple[Tensor, ...]) -> Tensor:
        img, masks, labels = sample
        semi = self(img)

        loss = loss_calculation(semi, labels, masks, self._criterion, self.hparams.cell_size)  # type: ignore
        self.train_loss(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log("train/loss", loss, on_step=False, on_epoch=True)

    def validation_step(self, sample: tuple[Tensor, ...]) -> None:
        img, masks, labels = sample
        semi = self(img)

        loss = loss_calculation(semi, labels, masks, self._criterion, self.hparams.cell_size)  # type: ignore
        self.val_loss(loss)

        heatmap = get_heatmap(semi)

        precision, recall = batch_precision_recall(
            heatmap,
            labels,
            masks,
            self.hparams.detection_threshold,  # type: ignore
            self.hparams.nms_dist,  # type: ignore
        )

        self.val_precision.update(precision)
        self.val_recall.update(recall)

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

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.hparams.learning_rate)  # type: ignore
        return optimizer
