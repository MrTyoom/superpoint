import os
import random
from pathlib import Path

import rootutils
import numpy as np
from PIL import Image
#from osgeo import gdal
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend, wrap_non_picklable_objects
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.common import make_dir  # noqa

random_state = np.random.RandomState(None)

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
    global random_state

    random_state = np.random.RandomState(cfg.seed)
    random.seed(cfg.seed)

    tiles = get_tiles(cfg)
    random.shuffle(tiles)

    images_dir = make_dir(cfg.images_dir, delete_if_exist=True)

    with parallel_backend("threading"):
        Parallel(n_jobs=os.cpu_count())(
            process(item, tiles[item], images_dir)
            for item in tqdm(range(len(tiles)), desc="tiles", leave=False)
        )


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.prepare_satellite_images)
