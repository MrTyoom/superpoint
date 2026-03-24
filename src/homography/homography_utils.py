"""Утилиты для работы с гомографиями"""

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from src.homography.apply import inv_warp_image_batch
from src.homography.generation import get_corners
from src.homography.transforms import perspective_transform
from src.types import Array, Tensor


EPS = 1e-7


def sample_homography(cfg: DictConfig, shape: Array, shift: int) -> tuple[Tensor, Tensor]:
    """
    Генерирует случайную гомографию
    """
    pts1, pts2 = get_corners(cfg.patch_ratio)
    pts2 = perspective_transform(cfg, pts2)

    pts1 *= shape[np.newaxis]
    pts2 *= shape[np.newaxis]

    inv_homography = cv2.getPerspectiveTransform(np.float32(pts1 + shift), np.float32(pts2 + shift))
    homography = np.linalg.inv(inv_homography)

    homography = torch.tensor(homography).float()
    inv_homography = torch.tensor(inv_homography).float()

    return homography, inv_homography


def compute_valid_mask(
    image_shape: Tensor | tuple[int, int] | torch.Size | list[int],
    inv_homography: Tensor,
    device: torch.device | str = "cpu",
    erosion_radius: int = 0,
) -> Tensor:
    """
    Вычисляет маску валидных пикселей после применения гомографии
    """
    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)

    batch_size = inv_homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)
    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode="nearest")
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()

    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2,) * 2)
        for el_ind in range(batch_size):
            mask[el_ind, :, :] = cv2.erode(mask[el_ind, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)


def homography_adaptation(heatmap, inv_homographies, mask_two_dim, device):
    heatmap *= mask_two_dim

    warped_heatmap = inv_warp_image_batch(heatmap, inv_homographies, device=device, mode="bilinear")
    warped_mask = inv_warp_image_batch(mask_two_dim, inv_homographies, device=device, mode="bilinear")

    sum_heatmap = torch.sum(warped_heatmap, dim=0)
    sum_mask = torch.sum(warped_mask, dim=0)

    return sum_heatmap / torch.clamp(sum_mask, min=EPS)
