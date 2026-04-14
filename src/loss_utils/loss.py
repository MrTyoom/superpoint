from typing import Callable

import torch
from torch.nn import BCELoss, Module
from torch.nn.functional import softmax

from src.loss_utils.sparse_loss import batch_descriptor_loss_sparse
from src.train_utils.d2s import SpaceToDepth
from src.types import Tensor


EPS = 1e-8


def detector_loss_calculation(
    semi: Tensor, labels: Tensor, masks: Tensor, criterion: Callable[..., Tensor], cell_size: int
) -> Tensor:
    labels_three_dim = get_labels(labels, cell_size, add_dustbin=True)
    mask_three_dim_flat = get_masks(masks, cell_size)

    loss_per_cell = criterion(softmax(semi, dim=1), labels_three_dim)
    loss = (loss_per_cell.sum(dim=1) * mask_three_dim_flat).sum()
    loss = loss / (mask_three_dim_flat.sum() + EPS)

    return loss


def get_labels(labels_two_dim: Tensor, cell_size: int, add_dustbin: bool) -> Tensor:
    batch_size, _, height, width = labels_two_dim.shape
    Hc, Wc = height // cell_size, width // cell_size
    space2depth = SpaceToDepth(8)

    labels_two_dim = space2depth(labels_two_dim)

    if add_dustbin:
        dustbin = labels_two_dim.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1] = 0
        labels_two_dim = torch.cat((labels_two_dim, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        # normalize
        dn = labels_two_dim.sum(dim=1)
        labels_two_dim = labels_two_dim.div(torch.unsqueeze(dn, 1))

    return labels_two_dim


def get_masks(masks_two_dim: Tensor, cell_size: int) -> Tensor:
    masks = get_labels(masks_two_dim, cell_size, add_dustbin=False)
    masks_three_dim_flat = torch.prod(masks, 1)

    return masks_three_dim_flat


class SuperPointLoss(Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.detector_loss = BCELoss(reduction="none")
        self.descriptor_loss = batch_descriptor_loss_sparse
        self.lambda_loss = cfg["lambda_loss"]
        self.cell_size = cfg["cell_size"]
        self.sparse_loss_cfg = cfg["sparse_loss"]

    def forward(self, semi, semi_w, desc, desc_w, labels_three_dim, labels_three_dim_w, masks, homo):  # noqa: WPS211
        loss_det = detector_loss_calculation(semi, labels_three_dim, masks[0], self.detector_loss, self.cell_size)
        loss_det_w = detector_loss_calculation(semi_w, labels_three_dim_w, masks[1], self.detector_loss, self.cell_size)

        loss_desc, _, _, _ = self.descriptor_loss(desc, desc_w, homo, cfg=self.sparse_loss_cfg)

        return loss_det + loss_det_w + self.lambda_loss * loss_desc
