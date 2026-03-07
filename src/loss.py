from typing import Callable

import torch
from torch.nn.functional import softmax
from torch.types import Tensor

from src.train_utils.d2s import SpaceToDepth

EPS = 1e-8


def loss_calculation(
    semi: Tensor, labels: Tensor, masks: Tensor, criterion: Callable[..., Tensor], cell_size: int
) -> int:
    labels_three_dim = get_labels(labels_two_dim=labels, cell_size=cell_size)
    mask_three_dim_flat = get_masks(masks_two_dim=masks, cell_size=cell_size)

    loss_per_cell = criterion(softmax(semi, dim=1), labels_three_dim)
    loss = (loss_per_cell.sum() * mask_three_dim_flat).sum()
    loss = loss / (mask_three_dim_flat.sum() + EPS)

    return loss


def labels_upscale(labels_two_dim: Tensor, cell_size: int) -> Tensor:
    batch_size, _, height, width = labels_two_dim.shape
    Hc, Wc = height // cell_size, width // cell_size
    space2depth = SpaceToDepth(8)

    labels_two_dim = space2depth(labels_two_dim)
    dustbin = labels_two_dim.sum(dim=1)
    dustbin = 1 - dustbin
    dustbin[dustbin < 1.0 - EPS] = 0

    labels_two_dim = torch.cat((labels_two_dim, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)

    # normalize
    dn = labels_two_dim.sum(dim=1)
    labels_two_dim = labels_two_dim.div(torch.unsqueeze(dn, 1))
    return labels_two_dim


def get_labels(labels_two_dim: Tensor, cell_size: int) -> Tensor:
    """
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
            labels: tensors[batch_size, 65, Hc, Wc]
    """
    return labels_upscale(labels_two_dim, cell_size)


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
    masks = labels_upscale(masks_two_dim, cell_size)
    masks_three_dim_flat = torch.prod(masks, 1)
    return masks_three_dim_flat
