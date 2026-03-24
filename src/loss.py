from typing import Callable

import numpy as np
import torch
import torchgeometry as tgm
from torch.nn.functional import softmax

from src.train_utils.d2s import SpaceToDepth
from src.train_utils.train_utils import to_numpy as train_to_numpy
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


def ext(img, pnt, wid):
    ll = pnt[1]
    lr = pnt[1] + wid
    rl = pnt[0]
    rr = pnt[0] + wid

    return img[ll:lr, rl:rr]


def extract_patch_from_points(heatmap, points, patch_size=5):
    if isinstance(heatmap, Tensor):  # type: ignore
        heatmap = train_to_numpy(heatmap)
    heatmap = heatmap.squeeze()

    pad_size = int(patch_size / 2)
    heatmap = np.pad(heatmap, pad_size, "constant")

    patches = []
    for idx in range(points.shape[0]):
        patch = ext(heatmap, points[idx, :].astype(int), patch_size)
        patches.append(patch)

    return patches


def soft_argmax_two_dim(patches, normalized_coordinates=True):
    """
    params:
        patches: (B, N, H, W)
    return:
        coor: (B, N, 2)  (x, y)

    """
    argm = tgm.contrib.SpatialSoftArgmax2d(normalized_coordinates=normalized_coordinates)
    coords = argm(patches)
    return coords


def do_log(patches):
    patches[patches < 0] = EPS
    patches_log = torch.log(patches)
    return patches_log


def norm_patches(patches):
    patch_size = patches.shape[-1]
    patches = patches.view(-1, 1, patch_size * patch_size)
    dk = torch.sum(patches, dim=-1).unsqueeze(-1) + EPS
    patches /= dk
    patches = patches.view(-1, 1, patch_size, patch_size)
    return patches


def soft_argmax_points(pts, heatmap, patch_size=5):
    """
    input:
        pts: tensor [N x 2]
    """
    pts = pts[0].transpose().copy()
    patches = extract_patch_from_points(heatmap, pts, patch_size=patch_size)

    patches = np.stack(patches)
    patches_torch = torch.tensor(patches, dtype=torch.float32).unsqueeze(0)
    patches_torch = norm_patches(patches_torch)
    patches_torch = do_log(patches_torch)

    dxdy = soft_argmax_two_dim(patches_torch, normalized_coordinates=False)

    points = pts
    points[:, :2] += dxdy.numpy().squeeze() - patch_size // 2

    pts_subpixel = [points.transpose().copy()]
    return pts_subpixel.copy()
