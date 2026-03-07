from pathlib import Path

import cv2
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.types import Tensor
from torch.utils.data import DataLoader, Dataset

from src.augmentation import PhotometricAugmentation
from src.homography.apply import filter_points, homography_scaling_torch, inv_warp_image, warp_points
from src.homography.homography_utils import compute_valid_mask, sample_homography

MAX_PIXEL = 255.0


def get_labels(pnts: Tensor, height: int, width: int) -> Tensor:
    labels = torch.zeros(height, width)
    pnts_round = pnts.round().long()

    max_coord_tensor = torch.tensor([[width - 1, height - 1]]).long()

    pnts_int = torch.min(pnts_round, max_coord_tensor)

    indices = (pnts_int[:, 1], pnts_int[:, 0])
    labels = labels.index_put_(indices, torch.ones(len(pnts_int)))
    return labels.unsqueeze(0)


class SyntheticDataset(Dataset):
    def __init__(self, files: list[Path], aug_cfg: DictConfig, mode: str = "train") -> None:
        super().__init__()

        self.image_files = files
        self.annot_files = [cur_file.with_suffix(".npy") for cur_file in files]

        self.mode = mode
        self.aug_cfg = aug_cfg

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        src_image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)
        src_points = np.load(self.annot_files[index])

        height, width = src_image.shape

        if self.mode == "train":
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

        if self.mode == "val":
            masks = torch.ones(height, width)
            masks = masks.unsqueeze(0)
        else:
            masks = compute_valid_mask(
                torch.tensor([height, width]),
                inv_homography=inv_homography,
                erosion_radius=self.aug_cfg.homographic.valid_border_margin,
            )

        labels = get_labels(warped_pts, height, width)

        sample = (warped_img, masks, labels)
        return sample


class Loader(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

        self.save_hyperparameters(logger=False)

    def train_dataloader(self) -> DataLoader:
        cfg = self.hparams.cfg
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=cfg.train_batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        cfg = self.hparams.cfg
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=cfg.val_batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_files, val_files = self.split_files()

            aug_cfg = self.hparams.cfg.augmentation

            self.train_dataset = SyntheticDataset(train_files, aug_cfg, "train")
            self.val_dataset = SyntheticDataset(val_files, aug_cfg, "val")

    def split_files(self) -> tuple[list[Path], list[Path]]:
        cfg = self.hparams.cfg

        files = list(Path(cfg.data_dir).glob("**/*.png"))
        files = [cur_file for cur_file in files if cur_file.with_suffix(".npy").exists()]

        rng = np.random.RandomState(cfg.seed)
        rng.shuffle(files)

        train_size = cfg.train_size
        num_train_files = int(len(files) * train_size)

        return files[:num_train_files], files[num_train_files:]
