import torch
from torch.types import Tensor

from src.train_utils.d2s import SpaceToDepth

EPS = 1e-8


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
