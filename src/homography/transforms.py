"""Функции для масштабирования, трансляции и поворота при генерации гомографий"""

import numpy as np
from numpy.random import randint, uniform
from omegaconf import DictConfig
from scipy.stats import truncnorm

from src.homography.generation import add_perspective
from src.types import Array


def calc_scales_and_center(cfg, pts: Array, std_trunc: int = 2) -> tuple[Array, Array]:
    """Вычисляет возможные масштабы и центр для масштабирования"""
    scales = truncnorm(-std_trunc, std_trunc, loc=1, scale=cfg.scaling_amplitude / 2).rvs(cfg.n_scales)
    scales = np.concatenate((np.array([1]), scales), axis=0)
    center = np.mean(pts, axis=0, keepdims=True)
    return scales, center


def get_scaling(cfg: DictConfig, pts: Array) -> Array:
    """Применяет случайное масштабирование к углам"""
    scales, center = calc_scales_and_center(cfg, pts)

    pts = (pts - center)[np.newaxis]
    pts = pts * np.expand_dims(scales, axis=(1, 2)) + center

    if cfg.allow_artifacts:
        valid = np.arange(cfg.n_scales)
    else:
        valid = (pts >= 0) * (pts < 1)
        valid = valid.prod(axis=1).prod(axis=1)
        valid = np.where(valid)[0]

    idx = valid[randint(valid.shape[0], size=1)].squeeze().astype(int)

    return pts[idx]


def get_translation(cfg: DictConfig, pts: Array) -> Array:
    """Применяет случайный сдвиг к углам"""
    t_min, t_max = np.min(pts, axis=0), np.min(1 - pts, axis=0)

    if cfg.allow_artifacts:
        t_min += cfg.translation_overflow
        t_max += cfg.translation_overflow

    x_rnd = uniform(-t_min[0], t_max[0], 1)
    y_rnd = uniform(-t_min[1], t_max[1], 1)

    return pts + np.array([x_rnd, y_rnd]).T


def calc_rotation(cfg: DictConfig, pts: Array):
    """Вычисляет матрицы поворота для всех возможных углов"""
    angles = np.linspace(-cfg.max_angle, cfg.max_angle, num=cfg.n_angles)
    angles = np.concatenate((angles, np.array([0])), axis=0)

    center = np.mean(pts, axis=0, keepdims=True)

    cs = np.cos(angles)
    sn = np.sin(angles)

    rot_mat = np.stack([cs, -sn, sn, cs], axis=1)
    rot_mat = np.reshape(rot_mat, [-1, 2, 2])

    return center, rot_mat


def get_rotation(cfg: DictConfig, pts: Array) -> Array:
    """Применяет случайный поворот к углам"""
    center, rot_mat = calc_rotation(cfg, pts)

    pts = np.matmul((pts - center)[np.newaxis], rot_mat) + center

    if cfg.allow_artifacts:
        valid = np.arange(cfg.n_angles)
    else:
        valid = (pts >= 0) * (pts < 1)
        valid = valid.prod(axis=1).prod(axis=1)
        valid = np.where(valid)[0]

    idx = valid[randint(valid.shape[0], size=1)].squeeze().astype(int)

    return pts[idx]


def perspective_transform(cfg: DictConfig, pts: Array) -> Array:
    """Последовательно применяет все трансформации к углам"""
    if cfg.perspective:
        pts = add_perspective(cfg, pts)
    if cfg.scaling:
        pts = get_scaling(cfg, pts)
    if cfg.translation:
        pts = get_translation(cfg, pts)
    if cfg.rotation:
        pts = get_rotation(cfg, pts)
    return pts
