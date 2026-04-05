"""Функции для генерации случайных гомографий"""

import numpy as np
from omegaconf import DictConfig
from scipy.stats import truncnorm

from src.types import Array


def get_corners(patch_ratio: int) -> tuple[Array, Array]:
    """Возвращает углы изображения и патча в нормализованных координатах"""
    # Corners of the output image
    pts1 = np.stack([[0, 0], [0, 1], [1, 1], [1, 0]], axis=0)

    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]])

    return pts1, pts2


def get_perspective_displacement(
    allow_artifacts: bool,
    perspective_amplitude_x: float,
    perspective_amplitude_y: float,
    margin: float,
    std_trunc: int = 2,
) -> Array:
    """Генерирует смещения для перспективного искажения"""
    if not allow_artifacts:
        perspective_amplitude_x = min(perspective_amplitude_x, margin)
        perspective_amplitude_y = min(perspective_amplitude_y, margin)

    perspective_displacement = truncnorm(-std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y / 2).rvs(1)
    h_displacement_left = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)
    h_displacement_right = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)

    return np.array(
        [
            [h_displacement_left, perspective_displacement],
            [h_displacement_left, -perspective_displacement],
            [h_displacement_right, perspective_displacement],
            [h_displacement_right, -perspective_displacement],
        ]
    ).squeeze()


def add_perspective(cfg: DictConfig, pts: Array) -> Array:
    """Добавляет перспективное искажение к углам"""
    displacement = get_perspective_displacement(
        cfg.allow_artifacts,
        cfg.perspective_amplitude_x,
        cfg.perspective_amplitude_y,
        margin=(1 - cfg.patch_ratio) / 2,
    )
    return pts + displacement
