import cv2
import numpy as np
from imgaug import augmenters as iaa
from numpy.random import rand, randint
from omegaconf import DictConfig

from src.types import TupleInt


def get_photometric_augmentation(cfg: DictConfig):
    change = cfg.random_brightness.max_abs_change
    aug = [iaa.Add((-change, change))]

    change = cfg.random_contrast.strength_range
    aug += [iaa.LinearContrast((change[0], change[1]))]

    change = cfg.additive_gaussian_noise.stddev_range
    aug += [iaa.AdditiveGaussianNoise(scale=(change[0], change[1]))]

    change = cfg.additive_speckle_noise.prob_range
    aug += [iaa.ImpulseNoise(p=(change[0], change[1]))]

    change = cfg.motion_blur.max_kernel_size

    if change > 3:
        change = randint(3, change)

    prob = 0.5
    aug += [iaa.Sometimes(prob, iaa.MotionBlur(change))]

    return iaa.Sequential(aug)


def get_ellipse_axes(dim: int) -> TupleInt:
    ax = int(max(rand() * dim, dim / 5))
    ay = int(max(rand() * dim, dim / 5))
    return ax, ay


def get_ellipse_center(img_shape: TupleInt, axes: TupleInt) -> TupleInt:
    max_rad = max(axes)
    x_cord = randint(max_rad, img_shape[1] - max_rad)
    y_cord = randint(max_rad, img_shape[0] - max_rad)
    return x_cord, y_cord


def blur_mask(kernel_size_range, mask):
    kernel_size = np.random.randint(*kernel_size_range)

    if (kernel_size % 2) == 0:
        kernel_size += 1

    mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)

    return mask


def shade_image(transparency_range, mask, img):
    transparency = np.random.uniform(*transparency_range)
    max_pixel = 255
    shaded = img * (1 - transparency * mask / max_pixel)
    shaded = np.clip(shaded, 0, max_pixel)
    img = shaded.astype(np.uint8)
    return img


def generate_mask(cfg: DictConfig, img_shape: TupleInt):
    mask = np.zeros(img_shape, np.uint8)
    color = 255
    max_angle = 360
    min_angle = 90

    for _ in range(cfg.nb_ellipses):
        axes = get_ellipse_axes(min(img_shape) // 4)

        cv2.ellipse(
            mask,
            get_ellipse_center(img_shape, axes),
            axes,
            np.random.rand() * min_angle,
            0,
            max_angle,
            color,  # type: ignore
            -1,
        )

    return mask


def additive_shade(cfg: DictConfig, mask, img):
    kernel_size = np.random.randint(*cfg.kernel_size_range)

    if (kernel_size % 2) == 0:
        kernel_size += 1

    mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    img = shade_image(cfg.transparency_range, mask, img)

    return img


class PhotometricAugmentation:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.aug = get_photometric_augmentation(cfg)

    def __call__(self, img):
        img = self.aug.augment_image(img)

        cfg = self.cfg.additive_shade
        mask = generate_mask(cfg, img.shape[:2])
        img = additive_shade(cfg, mask, img)

        return img
