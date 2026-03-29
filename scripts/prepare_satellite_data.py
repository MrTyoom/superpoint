import random
import shutil

import cv2
import numpy as np
import rootutils
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from pathlib import Path

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.common import make_dir

def get_random_crops(img, random_state, crop_h=360, crop_w=480, n=20):
    h, w, c = img.shape
    
    n_h = h // crop_h
    n_w = w // crop_w
    
    img_clipped = img[:n_h * crop_h, :n_w * crop_w, :]
    
    reshaped = img_clipped.reshape(n_h, crop_h, n_w, crop_w, c)
    crops = reshaped.transpose(0, 2, 1, 3, 4)
    crops = crops.reshape(-1, crop_h, crop_w, c)
    indices = random_state.choice(len(crops), n, replace=False)

    return crops[indices]

def divide_tiles(data_dir, cfg):
    dst_dir = Path(data_dir)
    
    make_dir(dst_dir)
    files = sorted([fp for fp in Path(cfg.data_dir).glob("**/*.jpg")])
    
    folder_idx = 0
    for ind in range(0, len(files), 4):
        batch = files[ind:ind + 4]
        
        new_folder = dst_dir / f"{folder_idx:06d}"
        make_dir(new_folder)
        
        for img_idx, img_path in enumerate(batch):
            shutil.copy(img_path, new_folder / f"{img_idx}.jpg")
        
        folder_idx += 1
        
def process_folder(file, dst_dir, random_state):
    file_num = int(file.name)
    
    batch = sorted(file.glob("*.jpg"))
    
    tile = {}
    
    for idx, img_path in enumerate(batch):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        crops = get_random_crops(img, random_state)

        tile[idx] = tuple(crops)
    
    ordered_tiles = [tile[i] for i in sorted(tile.keys())]
    
    for idx, crops_tuple in enumerate(zip(*ordered_tiles)):
        new_folder = dst_dir / f"{file_num * 20 + idx:06d}"
        make_dir(new_folder)
        
        for i, crop in enumerate(crops_tuple):
            cv2.imwrite(str(new_folder / f"{i}.jpg"), crop)
            
            
def select_all_crops(new_data_dir, old_data_dir, n_jobs=8, seed=42):
    dst_dir = Path(new_data_dir)
    make_dir(dst_dir)
    
    files = sorted(Path(old_data_dir).glob("*"))
    
    random_states = [
        np.random.RandomState(seed + i) for i in range(len(files))
    ]
    
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_folder)(file, dst_dir, random_states[i])
        for i, file in enumerate(tqdm(files))
    )


def main(cfg: DictConfig):
    random.seed(cfg.seed)
    
    tmp_dir = "tmp"
    make_dir(tmp_dir)
    
    divide_tiles(tmp_dir, cfg)
    select_all_crops(cfg.out_dir, tmp_dir)

    if Path(cfg.out_dir).exists():
        shutil.rmtree(tmp_dir)
    
if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.prepare_satellite_data)
