import cv2
import numpy as np
from numpy.random import randint, uniform
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.stats import truncnorm


def get_corners(patch_ratio: int) -> tuple[NDArray, NDArray]:
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
) -> NDArray:
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


def add_perspective(cfg: DictConfig, pts: NDArray) -> NDArray:
    displacement = get_perspective_displacement(
        cfg.allow_artifacts,
        cfg.perspective_amplitude_x,
        cfg.perspective_amplitude_y,
        margin=(1 - cfg.patch_ratio) / 2,
    )
    return pts + displacement


def calc_scales_and_center(cfg, pts: NDArray, std_trunc: int = 2) -> tuple[NDArray, NDArray]:
    scales = truncnorm(-std_trunc, std_trunc, loc=1, scale=cfg.scaling_amplitude / 2).rvs(cfg.n_scales)
    scales = np.concatenate((np.array([1]), scales), axis=0)
    center = np.mean(pts, axis=0, keepdims=True)
    return scales, center


def get_scaling(cfg: DictConfig, pts: NDArray) -> NDArray:
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


def get_translation(cfg: DictConfig, pts: NDArray) -> NDArray:
    t_min, t_max = np.min(pts, axis=0), np.min(1 - pts, axis=0)

    if cfg.allow_artifacts:
        t_min += cfg.translation_overflow
        t_max += cfg.translation_overflow

    x_rnd = uniform(-t_min[0], t_max[0], 1)
    y_rnd = uniform(-t_min[1], t_max[1], 1)

    return pts + np.array([x_rnd, y_rnd]).T


def calc_rotation(cfg: DictConfig, pts: NDArray):
    angles = np.linspace(-cfg.max_angle, cfg.max_angle, num=cfg.n_angles)
    angles = np.concatenate((angles, np.array([0])), axis=0)

    center = np.mean(pts, axis=0, keepdims=True)

    cs = np.cos(angles)
    sn = np.sin(angles)

    rot_mat = np.stack([cs, -sn, sn, cs], axis=1)
    rot_mat = np.reshape(rot_mat, [-1, 2, 2])

    return center, rot_mat


def get_rotation(cfg: DictConfig, pts: NDArray) -> NDArray:
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


def perspective_transform(cfg: DictConfig, pts: NDArray) -> NDArray:
    if cfg.perspective:
        pts = add_perspective(cfg, pts)
    if cfg.scaling:
        pts = get_scaling(cfg, pts)
    if cfg.translation:
        pts = get_translation(cfg, pts)
    if cfg.rotation:
        pts = get_rotation(cfg, pts)
    return pts


def sample_homography(cfg: DictConfig, shape: NDArray, shift: int) -> NDArray:
    pts1, pts2 = get_corners(cfg.patch_ratio)
    pts2 = perspective_transform(cfg, pts2)

    pts1 *= shape[np.newaxis]
    pts2 *= shape[np.newaxis]

    homography = cv2.getPerspectiveTransform(np.float32(pts1 + shift), np.float32(pts2 + shift))

    return homography
