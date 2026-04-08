import os
from pathlib import Path

import rootutils
import shutil
import random
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend, wrap_non_picklable_objects
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.common import make_dir

def get_tiles(cfg):
    crop_size = cfg.crop_size
    tiles = []

    for file in Path(cfg.satellite_dir).glob("**/*.jpg"):
        height = None
        width = None

        with Image.open(file) as fp:
            width, height = fp.size

        for y in range(0, height - crop_size, crop_size):
            for x in range(0, width - crop_size, crop_size):
                tiles.append([file, x, y, crop_size])

    return tiles


@delayed
@wrap_non_picklable_objects
def process(item, tile, data_dir):
    file, x, y, crop_size = tile
    
    with Image.open(file) as fp:
        area = (x, y, x + crop_size, y + crop_size)
        crop = fp.crop(area)
    
    out_dir = data_dir / str(item // 1_000).zfill(2)
    out_dir = make_dir(out_dir, delete_if_exist=False)
    
    crop.save(out_dir / f"{str(item).zfill(6)}.jpg")
    

def main(cfg):
    satellite_data_dir = Path(cfg.images_dir)
    
    images_dir = make_dir(satellite_data_dir / "tiles", delete_if_exist=True)
    tmp_dir = make_dir(satellite_data_dir / "tmp", delete_if_exist=True)
    
    tiles = get_tiles(cfg)

    with parallel_backend("threading"):
        Parallel(n_jobs=os.cpu_count())(
            process(item, tiles[item], images_dir)
            for item in tqdm(range(len(tiles)), desc="tiles", leave=False)
        )

    files = list(images_dir.glob("**/*.jpg"))
    
    random.shuffle(files)

    train_size = cfg.train_size
    num_train_files = int(len(files) * train_size)

    train_files = files[:num_train_files]
    test_files = files[num_train_files:]

    train_dir = make_dir(satellite_data_dir / "train")

    for n, img_file in enumerate(train_files):
        if n % 1000 == 0:
            out_dir = make_dir(train_dir / str(n // 1000).zfill(3))

        stem = out_dir / str(n % 1000).zfill(3)

        shutil.move(img_file, stem.with_suffix(img_file.suffix))
        
        npy_file = img_file.with_suffix(".npy")
        if npy_file.exists():
            shutil.move(npy_file, stem.with_suffix(".npy"))
    
    test_dir = make_dir(satellite_data_dir / "test")
    
    for n, img_file in enumerate(test_files):
        if n % 1000 == 0:
            out_dir = make_dir(test_dir / str(n // 1000).zfill(3))

        stem = out_dir / str(n % 1000).zfill(3)

        shutil.move(img_file, stem.with_suffix(img_file.suffix))
        
        npy_file = img_file.with_suffix(".npy")
        if npy_file.exists():
            shutil.move(npy_file, stem.with_suffix(".npy"))

    shutil.rmtree(images_dir)
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.prepare_satellite_images)
