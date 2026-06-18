from pathlib import Path

import cv2
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.augmentation import PhotometricAugmentation
from src.homography.apply import filter_points, homography_scaling_torch, inv_warp_image, warp_points
from src.homography.homography_utils import compute_valid_mask, sample_homography
from src.types import Tensor


MAX_PIXEL = 255.0
TRAIN_MODE = "train"


def get_labels(points: Tensor, height: int, width: int) -> Tensor:
    labels = torch.zeros(height, width)
    points_round = points.round().long()
    max_coord_tensor = torch.tensor([[width - 1, height - 1]]).long()
    points_int = torch.min(points_round, max_coord_tensor)
    indices = (points_int[:, 1], points_int[:, 0])
    labels = labels.index_put_(indices, torch.ones(len(points_int)))

    return labels.unsqueeze(0)


class MagicPointDataset(Dataset):
    def __init__(self, data_dir: Path, aug_cfg: DictConfig, mode: str = TRAIN_MODE) -> None:
        super().__init__()

        files = [fp for fp in Path(data_dir).glob("**/*.png") if fp.with_suffix(".npy").exists()]

        self.image_files = files
        self.annot_files = [fp.with_suffix(".npy") for fp in files]

        self.mode = mode
        self.aug_cfg = aug_cfg

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        src_image = cv2.imread(str(self.image_files[index]), cv2.IMREAD_GRAYSCALE)
        src_points = np.load(self.annot_files[index])

        height, width = src_image.shape

        if self.mode == TRAIN_MODE:
            aug = PhotometricAugmentation(self.aug_cfg.photometric)
            src_image = aug(src_image)

            homography, inv_homography = sample_homography(self.aug_cfg.homographic, np.array([2, 2]), shift=-1)
        else:
            homography = torch.eye(3)
            inv_homography = torch.eye(3)

        images = torch.from_numpy(src_image).float() / MAX_PIXEL

        warped_img = inv_warp_image(images.squeeze(), inv_homography, mode="bilinear").unsqueeze(0)

        homography_scaled = homography_scaling_torch(homography, height, width)

        pts_tensor = torch.from_numpy(src_points).float()
        warped_pts = warp_points(pts_tensor, homography_scaled)

        warped_pts = filter_points(warped_pts, torch.tensor([width, height]))

        if self.mode == TRAIN_MODE:
            masks = compute_valid_mask(
                torch.tensor([height, width]),
                inv_homography=inv_homography,
                erosion_radius=self.aug_cfg.homographic.valid_border_margin,
            )
        else:
            masks = torch.ones(height, width)
            masks = masks.unsqueeze(0)

        labels = get_labels(warped_pts, height, width)

        return warped_img, masks, labels


class MagicPointLoader(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        if stage == "fit":
            data_dir = Path(self.cfg.data_dir)
            aug_cfg = self.cfg.augmentation

            self.train_dataset = MagicPointDataset(data_dir / "train", aug_cfg, "train")
            self.val_dataset = MagicPointDataset(data_dir / "test", aug_cfg, "val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
