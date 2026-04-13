from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.homography.apply import filter_points, homography_scaling_torch, warp_points
from src.homography.homography_utils import compute_valid_mask
from src.transform import Augmentation


MAX_PIXEL = 255.0
TRAIN_MODE = "train"


def to_floatTensor(sample):
    return torch.tensor(sample).type(torch.FloatTensor)


def normalize_points(points, height, width):
    pts = points.clone()

    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    x_coords = (x_coords / width) * 2 - 1
    y_coords = (y_coords / height) * 2 - 1
    return pts


def denormalize_points(points, height, width):
    pts = points.clone()
    pts[:, 0] = (points[:, 0] + 1) * width / 2
    pts[:, 1] = (points[:, 1] + 1) * height / 2
    return pts


def points_to_two_dim(pnts, height, width):
    labels = np.zeros((height, width))
    pnts = pnts.int()
    labels[pnts[:, 1], pnts[:, 0]] = 1
    return labels


def get_center_crop_bounds(height, width, crop_h, crop_w):
    crop_h = min(crop_h, height)
    crop_w = min(crop_w, width)

    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    bottom = top + crop_h
    right = left + crop_w

    return left, right, top, bottom


def crop_tensor(tens, bounds):
    left, right, top, bottom = bounds
    return tens[:, top:bottom, left:right]


def crop_points(points, bounds, crop_h, crop_w):
    left, _, top, _ = bounds

    pts = points.clone()
    pts[:, 0] -= left
    pts[:, 1] -= top

    low_bound = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
    up_bound = (pts[:, 0] < crop_w) & (pts[:, 1] < crop_h)
    valid = low_bound & up_bound

    return pts[valid]


def crop_data(images, warped_img, mask, mask_w, pts, warped_pts, bounds, crop_h, crop_w):  # noqa: WPS211
    images = crop_tensor(images, bounds)
    warped_img = crop_tensor(warped_img, bounds)

    mask = crop_tensor(mask, bounds)
    mask_w = crop_tensor(mask_w, bounds)

    pts = crop_points(pts, bounds, crop_h, crop_w)
    warped_pts = crop_points(warped_pts, bounds, crop_h, crop_w)

    return (images, warped_img), (mask, mask_w), pts, warped_pts


def crop_homography(homography, left, top):
    homo = torch.tensor(
        [[1, 0, -left], [0, 1, -top], [0, 0, 1]],
        dtype=torch.float32,
    )
    homo = homo.unsqueeze(0)

    homo_inv = torch.tensor([[1, 0, left], [0, 1, top], [0, 0, 1]], dtype=torch.float32)  # noqa: WPS221
    homo_inv = homo_inv.unsqueeze(0)

    homo_crop = homo @ homography @ homo_inv
    inv_homo_crop = torch.linalg.inv(homo_crop)

    return homo_crop, inv_homo_crop


class SatelliteDataset(Dataset):
    def __init__(
        self, data_dir: Path, aug_cfg: DictConfig, mode: str = TRAIN_MODE, device: torch.device = "cpu"
    ) -> None:
        super().__init__()
        files = [fp for fp in Path(data_dir).glob("**/*.jpg") if fp.with_suffix(".npy").exists()]

        self.image_files = files
        self.annot_files = [fp.with_suffix(".npy") for fp in files]

        self.mode = mode
        self.aug_cfg = aug_cfg
        self.device = device

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict:
        src_image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)
        src_points = np.load(self.annot_files[index])

        H_full, W_full = src_image.shape

        images = torch.from_numpy(src_image).float().unsqueeze(0).unsqueeze(0) / MAX_PIXEL

        if self.mode == TRAIN_MODE:
            aug = Augmentation(self.aug_cfg)
            images = aug(images)

            homography = aug.homography
            inv_homography = aug.inv_homography

            warped_img = aug.warp(images)
        else:
            homography = torch.eye(3).unsqueeze(0)
            inv_homography = torch.eye(3).unsqueeze(0)
            warped_img = images.clone()

        images = images.squeeze(0)
        warped_img = warped_img.squeeze(0)

        if self.mode == TRAIN_MODE:
            mask = compute_valid_mask(
                torch.tensor([H_full, W_full]),
                inv_homography=inv_homography,
                erosion_radius=self.aug_cfg.valid_border_margin,
            )

            mask_w = compute_valid_mask(
                torch.tensor([H_full, W_full]),
                inv_homography=inv_homography,
                erosion_radius=self.aug_cfg.valid_border_margin,
            )
        else:
            mask = torch.ones(1, H_full, W_full)
            mask_w = torch.ones(1, H_full, W_full)

        pts_tensor = torch.from_numpy(src_points).float()[:, :2]

        homography_scaled = homography_scaling_torch(homography, H_full, W_full)

        pts_norm = normalize_points(pts_tensor, H_full, W_full)
        warped_pts_norm = warp_points(pts_norm, homography_scaled.squeeze(0))
        warped_pts = denormalize_points(warped_pts_norm, H_full, W_full)

        warped_pts = filter_points(warped_pts, torch.tensor([W_full, H_full]))

        crop_h = self.aug_cfg.crop_h
        crop_w = self.aug_cfg.crop_w

        bounds = get_center_crop_bounds(H_full, W_full, crop_h, crop_w)

        imgs, masks, pts_crop, warped_pts_crop = crop_data(
            images,
            warped_img,
            mask,
            mask_w,
            pts_tensor,
            warped_pts,
            bounds,
            crop_h,
            crop_w,
        )

        images, warped_img = imgs
        mask, mask_w = masks

        labels = points_to_two_dim(pts_crop, crop_h, crop_w)
        labels_two_dim = to_floatTensor(labels[np.newaxis, :, :])

        labels_w = points_to_two_dim(warped_pts_crop, crop_h, crop_w)
        labels_two_dim_w = to_floatTensor(labels_w[np.newaxis, :, :])

        left, _, top, _ = bounds
        homo_crop, inv_homo_crop = crop_homography(homography_scaled, left, top)

        return {
            "image": images,
            "warped_img": warped_img,
            "mask": mask,
            "mask_w": mask_w,
            "labels": labels_two_dim,
            "labels_w": labels_two_dim_w,
            "homo": homo_crop,
            "inv_homo": inv_homo_crop,
        }
