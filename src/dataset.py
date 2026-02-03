from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.augmentation import ImgAugTransform

class DataSet(Dataset):
    def __init__(self, files, aug_cfg, training):
        super().__init__()
        
        self.image_files = files
        self.annot_files = [f.with_suffix(".npy") for f in files]
        self.aug_cfg = aug_cfg
        self.training = training

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        src_image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)
        H, W = src_image.shape
        
        src_points = np.load(self.annot_files[index]) # x, y
        src_points = torch.from_numpy(src_points)

        # get_labels
        src_labels = torch.zeros(H, W)
        pnts_int = src_points.round().long()
        src_labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
        
        if self.training:
            augmentation = ImgAugTransform(self.aug_cfg.train.photometric)
            src_image = augmentation(src_image)
            
        return 0


def get_loaders(cfg):
    files = [f for f in Path(cfg.data_dir).glob("**/*.png") if f.with_suffix(".npy").exists()]
    
    np.random.shuffle(files)

    num_files = len(files)
    num_test_files = int(cfg.test_size * num_files)

    test_files = files[:num_test_files]
    train_files = files[num_test_files:]

    train_loader = DataLoader(
        DataSet(train_files, cfg.augmentation, training=True), 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=True, 
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        #prefetch_factor=2
    )

    test_loader = DataLoader(
        DataSet(test_files, cfg.augmentation, training=False), 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=False, 
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        #prefetch_factor=2
    )
    
    return train_loader, test_loader
