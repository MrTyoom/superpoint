from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.homography.apply import filter_points, homography_scaling_torch, warp_points
from src.homography.homography_utils import compute_valid_mask
from src.loss_utils.loss import get_labels, get_masks
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

        height, width = src_image.shape
        images = torch.from_numpy(src_image).float().unsqueeze(0).unsqueeze(0) / MAX_PIXEL

        if self.mode == TRAIN_MODE:
            aug = Augmentation(self.aug_cfg)

            images = aug(images)

            homography = aug.homography
            inv_homography = aug.inv_homography

            warped_img = aug.warp(images)
        else:
            homography = torch.eye(3)
            inv_homography = torch.eye(3)
            warped_img = images.clone()

        images = images.squeeze(0)
        warped_img = warped_img.squeeze(0)

        homography = homography_scaling_torch(homography, height, width)

        pts_tensor = torch.from_numpy(src_points).float()
        warped_pts = warp_points(pts_tensor[:, :2], homography.squeeze(0))
        warped_pts = filter_points(warped_pts, torch.tensor([width, height]))

        if self.mode == TRAIN_MODE:
            mask = compute_valid_mask(
                torch.tensor([height, width]),
                inv_homography=inv_homography,
                erosion_radius=self.aug_cfg.valid_border_margin,
            )

            mask_w = compute_valid_mask(
                torch.tensor([height, width]),
                inv_homography=homography,
                erosion_radius=self.aug_cfg.valid_border_margin,
            )
        else:
            mask = torch.ones(1, height, width)
            mask_w = torch.ones(1, height, width)

        labels = get_labels(pts_tensor[:, :2], height, width)
        labels_w = get_labels(warped_pts, height, width)

        mask_flat = get_masks(mask, self.aug_cfg.cell_size)
        mask_flat_w = get_masks(mask_w, self.aug_cfg.cell_size)

        return {
            "image": images,
            "warped_img": warped_img,
            "mask": mask_flat,
            "mask_w": mask_flat_w,
            "labels": labels,
            "labels_w": labels_w,
            "homo": homography,
            "inv_homo": inv_homography,
        }
