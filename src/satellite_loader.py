from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.homography.apply import filter_points, homography_scaling_torch, warp_points
from src.homography.homography_utils import compute_valid_mask
from src.train_utils.crop_utils import crop_data, crop_homography, get_center_crop_bounds
from src.train_utils.train_utils import as_float_tensor, denormalize_points, normPts, points_to_two_dim
from src.transform import Augmentation


MAX_PIXEL = 255.0
TRAIN_MODE = "train"


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

        pts_norm = normPts(pts_tensor, W_full)
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
        labels_two_dim = as_float_tensor(labels[np.newaxis, :, :])

        labels_w = points_to_two_dim(warped_pts_crop, crop_h, crop_w)
        labels_two_dim_w = as_float_tensor(labels_w[np.newaxis, :, :])

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
