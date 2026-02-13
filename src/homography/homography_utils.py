"""Утилиты для работы с гомографиями"""

import cv2
import numpy as np
import torch
from apply import inv_warp_image_batch
from generation import get_corners
from numpy.typing import NDArray
from omegaconf import DictConfig
from transforms import perspective_transform


def sample_homography(cfg: DictConfig, shape: NDArray, shift: int) -> NDArray:
    """
    Генерирует случайную гомографию
    """
    pts1, pts2 = get_corners(cfg.patch_ratio)
    pts2 = perspective_transform(cfg, pts2)

    pts1 *= shape[np.newaxis]
    pts2 *= shape[np.newaxis]

    homography = cv2.getPerspectiveTransform(np.float32(pts1 + shift), np.float32(pts2 + shift))
    homography = np.linalg.inv(homography)

    return homography


def compute_valid_mask(image_shape, inv_homography, device="cpu", erosion_radius=0):
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
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)
