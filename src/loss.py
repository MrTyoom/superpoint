from typing import Callable

import torch
from torch.nn.functional import softmax

from src.train_utils.d2s import SpaceToDepth
from src.types import Tensor

EPS = 1e-8


def loss_calculation(
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
