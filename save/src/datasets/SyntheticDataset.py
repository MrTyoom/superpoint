import random
import torch
import sys
import numpy as np
import torch.utils.data as data

from imageio import imread
from pathlib import Path
from omegaconf import OmegaConf


sys.path.append(str(Path(__file__).parent.parent))


from settings import DATA_PATH
from utils.homographies import sample_homography_np as sample_homography
from utils.utils import (compute_valid_mask, inv_warp_image, warp_points,
                         filter_points)
from utils.utils import homography_scaling_torch as homography_scaling
from utils.photometric import ImgAugTransform, customizedTransform

def load_as_float(path):
    return imread(path).astype(np.float32) / 255

class SyntheticDataset(data.Dataset):
    def __init__(self,
        seed=None,
        task="train",
        sequence_length=3,
        transform=None,
        target_transform=None,
        warp_input=False,
        **config
    ):
        self.seed = seed
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_dtype(torch.float32)
  
        
        if seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                self.cuda_generator = torch.Generator(device='cuda')
                self.cuda_generator.manual_seed(self.seed)
            else:
                self.cuda_generator = None
                
        self.config = config
        
        self.transform = transform
        self.sample_homography = sample_homography
        self.compute_valid_mask = compute_valid_mask
        self.inv_warp_image = inv_warp_image
        self.warp_points = warp_points
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform
        
        ######
        self.enable_photo_train = self.config["aug"]["photometric"]["enable"]
        self.enable_homo_train = self.config["aug"]["homographic"]["enable"]
        self.enable_homo_val = False
        self.enable_photo_val = False
        ######
        
        self.task = task
        self.action = 'training' if task == 'train' else 'validation'
        self.data = self.split_data()[0 if self.action == 'train' else 1]
        
    @staticmethod
    def get_labels(pnts, H, W, device=None):
        pnts = pnts.to(device)
        
        labels = torch.zeros(H, W, device=device)
        pnts_int = torch.min(pnts.round().long(), 
                           torch.tensor([[W - 1, H - 1]], device=device).long())
        labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
        return labels
    
    def get_label_res(self, H, W, pnts, device=None):
        """Создание карты субпиксельных смещений"""
        labels_res = torch.zeros(H, W, 2, device=device)
        pnts_int = pnts.round().long()
        
        pnts = pnts.to(device)
        pnts_int = pnts_int.to(device)
        
        labels_res[pnts_int[:, 1], pnts_int[:, 0], :] = pnts - pnts.round()
        labels_res = labels_res.transpose(1, 2).transpose(0, 1)
        return labels_res
        
    def imgPhotometric(self, img):
        augmentation = self.ImgAugTransform(**self.config["aug"])
        img = img[:, :, np.newaxis]
        img = augmentation(img)
        cusAug = self.customizedTransform()
        #img = cusAug(img, **self.config["aug"])
        img = cusAug(img)
        return img
    
    def split_data(self):
        random_state = np.random.RandomState(self.seed)

        base_dir = Path(DATA_PATH , 'data', 'synthetic_dataset')
        all_primitives = list(self.config['primitives'].keys())

        all_files = []

        for primitive in all_primitives:
            for el in range(20):
                dir_path = base_dir / Path(primitive, f"00{el:02d}")
                for ind in range(1000):
                    img_path = dir_path / f'{el * 1000 + ind:06d}.png'
                    pts_path = dir_path / f'{el * 1000 + ind:06d}.npy'
                    all_files.append((img_path, pts_path))
                    
        total_files = len(all_files)
        train_count = int(self.config['train_size'] * total_files)

        all_indices = np.arange(total_files)  
        train_ind = random_state.choice(all_indices, size=train_count, replace=False)
        train_ind = set(train_ind)
        val_ind = set(all_indices) - train_ind

        train_data = [all_files[i] for i in train_ind]
        val_data = [all_files[i] for i in val_ind]

        return train_data, val_data
    
    def __getitem__(self, index):
        img_path, pts_path = self.data[index]
        
        img = load_as_float(img_path)
        pts = np.load(pts_path)
        
        H, W = img.shape[0], img.shape[1]
        
        
        if (self.config["aug"]["photometric"]["enable_train"] and self.action == "training") or (
            self.config["aug"]["photometric"]["enable_val"] and self.action == "validation"
        ):
            img = self.imgPhotometric(img)
        
        apply_homography = (self.config["aug"]["homographic"]["enable_train"] and self.action == "training")\
                            or (self.config["aug"]["homographic"]["enable_val"] and self.action == "validation")
        
        if apply_homography:
            homography = self.sample_homography(
                np.array([2, 2]),
                shift=-1
            )

            homography = np.linalg.inv(homography)
            homography = torch.tensor(homography).float()
            inv_homography = homography.inverse()
            
            device = self.device
            
            img_tensor = torch.from_numpy(img).float()
            warped_img = self.inv_warp_image(img_tensor.squeeze(), inv_homography, mode="bilinear")
            warped_img = warped_img.squeeze().numpy()
            warped_img = warped_img[:, :, np.newaxis]

            homography_scaled = homography_scaling(homography, H, W)
            
            pts_tensor = torch.from_numpy(pts).float()
            warped_pts = self.warp_points(pts_tensor, homography_scaled)
            warped_pts = filter_points(warped_pts, torch.tensor([W, H]))
            
            if self.transform is not None:
                warped_img = self.transform(warped_img)
            
            valid_mask = self.compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self.config["aug"]["homographic"]["valid_border_margin"],
            ).to(device)
            
            # Используем правильный device для меток
            labels_2D = self.get_labels(warped_pts, H, W, device=device)
            warped_labels_res = self.get_label_res(H, W, warped_pts, device=device)
            
            img_tensor = torch.from_numpy(img).float()
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            
            warped_img_tensor = warped_img.clone()
            if warped_img_tensor.dim() == 2:
                warped_img_tensor = warped_img_tensor.unsqueeze(0)
            elif warped_img_tensor.dim() == 3 and warped_img_tensor.shape[2] == 1:
                warped_img_tensor = warped_img_tensor.permute(2, 0, 1)
            warped_img_tensor = warped_img_tensor.to(device)
        
        sample = (
            img_tensor,
            labels_2D,
            valid_mask,
            warped_img_tensor,
            warped_labels_res,
            homography
        )
        # [1, 240, 320]
        
        return sample
    
    def __len__(self):
        return len(self.data)