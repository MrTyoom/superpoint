from pathlib import Path

import cv2
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class SuperPointDataset(Dataset):
    def __init__(self, images_dir: Path, points_dir: Path, aug_cfg: DictConfig) -> None:
        super().__init__()

        self.images_files = []
        self.points_files = []

        for img_file in images_dir.glob("**/*.jpg"):
            pnt_file = points_dir / img_file.relative_to(images_dir).with_suffix(".npy")

            if pnt_file.exists():
                self.images_files.append(img_file)
                self.points_files.append(pnt_file)

        self.aug_cfg = aug_cfg

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index: int):
        image = cv2.imread(str(self.images_files[index]), cv2.IMREAD_GRAYSCALE)

        blur_size = self.aug_cfg.blur_size
        image = cv2.blur(image, (blur_size, blur_size))

        points_and_confidences = np.load(self.points_files[index])  # x, y, confidence

        points = points_and_confidences[:, :2]
        confidence = points_and_confidences[:, 2]

        good_points_mask = confidence > self.aug_cfg.confidence_threshold
        points = points[good_points_mask]

        labels = np.zeros_like(image)
        xy = points.round().astype(int)
        labels[xy[:, 1], xy[:, 0]] = 1

        return image, labels


class SuperPointLoader(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        if stage == "fit":
            images_dir = Path(self.cfg.images_dir)
            points_dir = Path(self.cfg.points_dir)

            aug_cfg = self.cfg.augmentation

            self.train_dataset = SuperPointDataset(images_dir / "train", points_dir / "train", aug_cfg)
            self.val_dataset = SuperPointDataset(images_dir / "test", points_dir / "test", aug_cfg)

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
