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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(device)
        np.random.seed(self.seed)
        random.seed(self.seed)

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
        
    def imgPhotometric(self, img):
        """

        :param img:
            numpy (H, W)
        :return:
        """
        augmentation = self.ImgAugTransform(**self.config["augmentation"])
        img = img[:, :, np.newaxis]
        img = augmentation(img)
        cusAug = self.customizedTransform()
        img = cusAug(img, **self.config["augmentation"])
        return img
    
    def get_labels(pnts, H, W):
        labels = torch.zeros(H, W)
        pnts_int = torch.min(pnts.round().long(), torch.tensor([[W - 1, H - 1]]).long())
        labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
        return labels
    

    def get_label_res(self, H, W, pnts):
        """Создание карты субпиксельных смещений"""
        labels_res = torch.zeros(H, W, 2)
        pnts_int = pnts.round().long()
        labels_res[pnts_int[:, 1], pnts_int[:, 0], :] = pnts - pnts.round()
        labels_res = labels_res.transpose(1, 2).transpose(0, 1)
        return labels_res
    
    def split_data(self):
        random_state = np.random.RandomState(self.seed)

        base_dir = Path(DATA_PATH , 'data', 'synthetic_dataset')
        all_primitives = list(self.config['primitives'].keys())

        all_files = []

        for primitive in all_primitives:
            for el in range(20):
                dir_path = base_dir / Path(primitive, f"{el:02d}")
                for ind in range(1000):
                    img_path = dir_path / f'00{ind:04d}.png'
                    pts_path = dir_path / f'00{ind:04d}.npy'
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
        
        
        # labels_2D = self.get_labels(pts, H, W)
        # labels_2D = torch.from_numpy(labels_2D)
        
        
        if (self.config["aug"]["photometric"]["enable_train"] and self.action == "training") or (
            self.config["aug"]["photometric"]["enable_val"] and self.action == "validation"
        ):
            img = self.imgPhotometric(img)
        else:
            pass
        
        apply_homography = (self.config["aug"]["homographic"]["enable_train"] and self.action == "training")\
                            or (self.config["aug"]["homographic"]["enable_val"] and self.action == "validation")
        # if not apply_homography:
        #     img = img[:, :, np.newaxis]
           
        #     if self.transform is not None:
        #         img = self.transform(img)
                
        #     img = torch.from_numpy(img)
        #     valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
            
        #     labels_res = self.get_label_res(H, W, pts)
           
        if apply_homography:
            homography = self.sample_homography(
                np.array([2, 2]),
                shift=-1
            )

            homography = np.linalg.inv(homography)
            homography = torch.tensor(homography).float()
            inv_homography = homography.inverse()
            img = torch.from_numpy(img)
            warped_img = self.inv_warp_image(img.squeeze(), inv_homography, mode="bilinear")
            warped_img = warped_img.squeeze().numpy()
            warped_img = warped_img[:, :, np.newaxis]

            warped_pts = self.warp_points(pts, homography_scaling(homography, H, W))
            warped_pts = filter_points(warped_pts, torch.tensor([W, H]))

            if self.transform is not None:
                warped_img = self.transform(warped_img)
            
            valid_mask = self.compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self.config["aug"]["homographic"]["valid_border_margin"],
            )  # can set to other value

            labels_2D = self.get_labels(warped_pts, H, W)
            warped_labels_res = self.get_label_res(H, W, warped_pts)
            
        sample = (
            img,
            labels_2D,
            homography,
            valid_mask,
            warped_img,
            warped_pts,
            warped_labels_res
        )
        # else:
        #     sample = (
        #         img, 
        #         valid_mask,
        #         label_res
        #     )
        

        return sample
    
    def __len__(self):
        return len(self.data)
    

  
if __name__ == "__main__":
    params_path = Path(__file__).parent.parent / 'params.yaml'
    cfg = OmegaConf.load(str(params_path))
    
    training_cfg = cfg['prepare_synthetic_dataset']
    dataset = SyntheticDataset(**training_cfg)
    
    train_data, val_data = dataset.split_data()
    print(train_data[0], val_data[0])