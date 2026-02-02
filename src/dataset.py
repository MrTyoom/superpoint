from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DataSet(Dataset):
    def __init__(self, files, aug, training):
        super().__init__()
        
        self.image_files = files
        self.annot_files = [f.with_suffix(".npy") for f in files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return 0


def get_loaders(cfg):
    files = [f for f in Path(cfg.data_dir).glob("**/*.png") if f.with_suffix(".npy").exists()]
    
    np.random.shuffle(files)

    num_files = len(files)
    num_test_files = int(cfg.test_size * num_files)

    test_files = files[:num_test_files]
    train_files = files[num_test_files:]

    train_loader = DataLoader(
        DataSet(train_files, cfg.train_aug, training=True), 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=True, 
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2
    )

    test_loader = DataLoader(
        DataSet(test_files, aug=cfg.train_aug, training=False), 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=False, 
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2
    )
    
    return train_loader, test_loader
