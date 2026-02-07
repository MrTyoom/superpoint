from pathlib import Path

import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split


class DataSet(Dataset):
    def __init__(self, files):
        super().__init__()

        self.image_files = files
        self.annot_files = [f.with_suffix(".npy") for f in files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return 0


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
            train_dataset, val_dataset = self.split_dataset()

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

    def split_dataset(self):
        files = [
            f
            for f in Path(self.hparams.cfg.data_dir).glob("**/*.png")
            if f.with_suffix(".npy").exists()
        ]

        dataset = DataSet(files)

        dataset_size = len(dataset)
        train_size = int(dataset_size * self.hparams.cfg.train_size)

        return random_split(dataset, [train_size, dataset_size - train_size])
