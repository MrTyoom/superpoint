import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import torch.nn.functional as F
from dvc.api import DVCFileSystem
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

root = rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.synthetic_loader import Loader, SyntheticDataset
from src.train_magicpoint import MagicPointLightning

output_dir = root / "tmp" / "train_magic_point_samples"


os.makedirs(output_dir, exist_ok=True)


def extract_keypoints(semi, conf_thresh=0.5):
    B, C, Hc, Wc = semi.shape
    
    semi = semi.permute(0, 2, 3, 1)
    heatmap = F.softmax(semi, dim=-1)[..., :-1]

    heatmap = heatmap.sum(dim=-1)

    hmap = heatmap[0]
    ys, xs = (hmap > conf_thresh).nonzero(as_tuple=True)
    ys *= 8
    xs *= 8    
    return xs.cpu().numpy(), ys.cpu().numpy()

def main(cfg):
    cfg_train = cfg["train_magicpoint"]
    
    fs = DVCFileSystem(
        repo=".",
        rev="as/train_magic_point",
        remote="storage",
    )

    model_dvc_path = "models/magic_point.ckpt"

    local_model_path = "/tmp/magic_point.ckpt"
    with fs.open(model_dvc_path, "rb") as f_src:
        with open(local_model_path, "wb") as f_dst:
            f_dst.write(f_src.read())

    model = MagicPointLightning(cfg_train)
    ckpt = torch.load(local_model_path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    datamodule = Loader(cfg_train)
    datamodule.setup(stage="fit")
    
    test_dataloader = datamodule.val_dataloader()
    
    for batch_idx, batch in enumerate(test_dataloader):
        img, mask, labels = batch
        with torch.no_grad():
            output = model(img)
        
        batch_size = img.shape[0]
        for i in range(batch_size):
            xs, ys = extract_keypoints(output[i:i+1], conf_thresh=0.01)
        
            img_np = img[i].permute(1, 2, 0).cpu().numpy()

            label_mask = labels[i, 0]
            y_orig, x_orig = (label_mask > 0).nonzero(as_tuple=True)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img_np, cmap='gray')
            plt.scatter(xs, ys, c='r', s=10, label='predicted')
            plt.scatter(x_orig, y_orig, c='b', s=10, label='ground_truth')
            plt.title(f"Image {batch_idx * batch_size + i}")
            plt.legend()
            plt.axis('off')
            
            save_path = os.path.join(output_dir, f"image_{batch_idx * batch_size + i}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        break

            
if __name__ == '__main__':
    cfg = OmegaConf.load("params.yaml")
    main(cfg)
