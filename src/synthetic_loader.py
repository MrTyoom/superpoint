from pathlib import Path

import cv2
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from src.augmentation import PhotometricAugmentation
from src.homography.apply import filter_points, homography_scaling_torch, inv_warp_image, warp_points
from src.homography.homography_utils import compute_valid_mask, sample_homography


def get_labels(pnts, H, W, device=None):
    labels = torch.zeros(H, W, device=device)
    pnts_round = pnts.round().long().to(device)

    max_coord_tensor = torch.tensor([[W - 1, H - 1]], device=device).long()

    pnts_int = torch.min(pnts_round, max_coord_tensor)

    indices = (pnts_int[:, 1], pnts_int[:, 0])
    labels = labels.index_put_(indices, torch.ones(len(pnts_int), device=device))
    return labels


class DataSet(Dataset):
    def __init__(self, files: list[Path], aug_cfg: DictConfig | None = None, device: torch.device = None):
        super().__init__()

        self.image_files = files
        self.annot_files = [f.with_suffix(".npy") for f in files]

        self.aug_cfg = aug_cfg
        self.device = "cpu"
        self.transform = ToTensor()

    def get_label_res(self, H, W, pnts, device=None):
        """Создание карты субпиксельных смещений"""
        pnts = pnts.to(device)
        labels_res = torch.zeros(2, H, W, device=device)  # (2, H, W)

        pnts_int = pnts.round().long()
        max_coords = torch.tensor([W - 1, H - 1], device=device).long()
        pnts_int = torch.min(pnts_int, max_coords)

        offsets = pnts - pnts.round()  # (N, 2)
        y_indices = pnts_int[:, 1]
        x_indices = pnts_int[:, 0]

        # Первый канал (dx)
        labels_res[0].index_put_(
            (y_indices, x_indices),
            offsets[:, 0],  # dx
        )

        # Второй канал (dy)
        labels_res[1].index_put_(
            (y_indices, x_indices),
            offsets[:, 1],  # dy
        )
        return labels_res  # (2, H, W)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        src_image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)
        src_points = np.load(self.annot_files[index])  # x, y

        H, W = src_image.shape

        if self.aug_cfg is None:
            # validation == without augmentation
            homography = torch.eye(3)
            inv_homography = homography.clone()

        else:
            # train
            aug = PhotometricAugmentation(self.aug_cfg.photometric)
            src_image = aug(src_image)

            inv_homography = sample_homography(self.aug_cfg.homographic, np.array([2, 2]), shift=-1)  # H_inv
            inv_homography = torch.tensor(inv_homography).float()

            homography = np.linalg.inv(inv_homography)  # H
            homography = torch.tensor(homography).float()

        img_tensor = torch.from_numpy(src_image).float()

        warped_img = inv_warp_image(img_tensor.squeeze(), inv_homography, mode="bilinear")
        warped_img = warped_img.squeeze().numpy()
        warped_img = warped_img[:, :, np.newaxis]

        homography_scaled = homography_scaling_torch(homography, H, W)

        pts_tensor = torch.from_numpy(src_points).float()
        warped_pts = warp_points(pts_tensor, homography_scaled)

        warped_pts = filter_points(warped_pts, torch.tensor([W, H]))

        warped_img = self.transform(warped_img)

        if self.aug_cfg is None:
            valid_mask = torch.ones(H, W).to(self.device)
        else:
            valid_mask = compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                device=self.device,
                erosion_radius=self.aug_cfg.homographic.valid_border_margin,
            ).to(self.device)

        labels_two_dim = get_labels(warped_pts, H, W, device=self.device)
        warped_labels_res = self.get_label_res(H, W, warped_pts, device=self.device)

        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(self.device)

        warped_img_tensor = warped_img.clone()
        if warped_img_tensor.dim() == 2:
            warped_img_tensor = warped_img_tensor.unsqueeze(0)
        elif warped_img_tensor.dim() == 3 and warped_img_tensor.shape[2] == 1:
            warped_img_tensor = warped_img_tensor.permute(2, 0, 1)
        warped_img_tensor = warped_img_tensor.to(self.device)

        sample = (img_tensor, labels_two_dim, valid_mask, warped_img_tensor, warped_labels_res, homography)
        # [1, 120, 160]

        return sample


class Loader(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

        self.save_hyperparameters(logger=False)
        self.device = "cpu"

        self.generator = torch.Generator("cpu")
        self.generator.manual_seed(self.hparams.cfg.seed)

    def train_dataloader(self) -> DataLoader:
        cfg = self.hparams.cfg
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=cfg.train_batch_size,
            num_workers=cfg.num_workers,
            generator=self.generator,
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

            self.train_dataset = DataSet(train_files, aug_cfg, self.device)
            self.val_dataset = DataSet(val_files, None, device=self.device)

    def split_files(self):
        cfg = self.hparams.cfg

        files = Path(cfg.data_dir).glob("**/*.png")
        files = [f for f in files if f.with_suffix(".npy").exists()]

        rng = np.random.RandomState(cfg.seed)
        rng.shuffle(files)

        train_size = cfg.train_size
        num_train_files = int(len(files) * train_size)

        return files[:num_train_files], files[num_train_files:]
