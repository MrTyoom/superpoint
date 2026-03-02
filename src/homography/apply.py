"""Функции для применения гомографий к изображениям и точкам"""

import torch
from torch.nn.functional import grid_sample
from torch.types import Tensor

TWO = 2.0


def inv_warp_image_batch(
    img: Tensor, mat_homo_inv: Tensor, device: torch.device | str = "cpu", mode: str = "bilinear"
) -> Tensor:
    """
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    """
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1, 1, img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1, 3, 3)

    batch, _, height, width = img.shape

    vert_space = torch.linspace(-1, 1, width)
    height_space = torch.linspace(-1, 1, height)

    grid = torch.meshgrid(vert_space, height_space, indexing="ij")
    coor_cells = torch.stack(grid, dim=2)
    coor_cells = coor_cells.transpose(0, 1)

    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([batch, height, width, 2])
    src_pixel_coords = src_pixel_coords.float()

    warped_img = grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img


def inv_warp_image(
    img: Tensor, mat_homo_inv: Tensor, device: torch.device | str = "cpu", mode: str = "bilinear"
) -> Tensor:
    """
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [H, W]
    """
    warped_img = inv_warp_image_batch(img, mat_homo_inv, device, mode)
    return warped_img.squeeze()


def warp_points(points: Tensor, homographies: Tensor, device: torch.device | str = "cpu") -> Tensor:
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies

    batch_size = homographies.shape[0]

    # homogen coords
    all_ones = torch.ones((points.shape[0], 1)).to(device)
    points = torch.cat((points.float(), all_ones), dim=1)
    points = points.to(device)

    # (B, 3, 3) -> (Bx3, 3)
    homographies = homographies.view(batch_size * 3, 3)
    warped_points = homographies @ points.transpose(0, 1)

    # (B×3, N) -> (B, 3, N) -> (B, N, 3)
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)

    # back to decart coords
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]

    return warped_points[0, :, :] if no_batches else warped_points


def homography_scaling_torch(homography: Tensor, height: int, width: int) -> Tensor:
    row1 = torch.tensor([TWO / width, 0, -1])
    row2 = torch.tensor([0, TWO / height, -1])
    row3 = torch.tensor([0, 0, 1.0])

    trans = torch.stack([row1, row2, row3])
    homography = trans.inverse() @ homography @ trans
    return homography


def filter_points(points: Tensor, shape: Tensor, return_mask: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape - 1)
    mask = torch.prod(mask, dim=-1) == 1
    if return_mask:
        return points[mask], mask
    return points[mask]
