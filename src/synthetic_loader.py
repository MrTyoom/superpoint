from pathlib import Path

import cv2
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.augmentation import PhotometricAugmentation


class DataSet(Dataset):
    def __init__(self, files: list[Path], aug_cfg: DictConfig | None = None):
        super().__init__()

        self.image_files = files
        self.annot_files = [f.with_suffix(".npy") for f in files]

        self.aug_cfg = aug_cfg

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        src_image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)

        # H, W = src_image.shape
        # src_points = np.load(self.annot_files[index])  # x, y
        # src_points = torch.from_numpy(src_points)
        # get_labels
        # src_labels = torch.zeros(H, W)
        # pnts_int = src_points.round().long()
        # src_labels[pnts_int[:, 1], pnts_int[:, 0]] = 1

        if self.aug_cfg is not None:
            aug = PhotometricAugmentation(self.aug_cfg.photometric)
            aug_image = aug(src_image)

        return aug_image


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
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        cfg = self.hparams.cfg
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_files, val_files = self.split_files()

            aug_cfg = self.hparams.cfg.augmentation

            self.train_dataset = DataSet(train_files, aug_cfg)
            self.val_dataset = DataSet(val_files)

    def split_files(self):
        cfg = self.hparams.cfg
        files = [
            f
            for f in Path(cfg.data_dir).glob("**/*.png")
            if f.with_suffix(".npy").exists()
        ]
        np.random.shuffle(files)

        train_size = cfg.train_size
        num_train_files = int(len(files) * train_size)

        return files[:num_train_files], files[num_train_files:]
