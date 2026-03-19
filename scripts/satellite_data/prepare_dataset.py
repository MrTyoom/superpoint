import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.common import make_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_files(data_dir: Path):
    if not data_dir.exists():
        return []

    files = list(data_dir.glob("**/*.jpg")) + list(data_dir.glob("**/*.png"))

    return files


def read_image(image_file: Path):
    image = Image.open(image_file)

    if image_file.suffix == ".png":
        image = image.convert("L")

    return np.asarray(image)


def process_layer(item: tuple[tuple[int, int], list[Path]], cfg: DictConfig, dataset_path: Path, task_index: int, max_len: int):
    (x, y), files = item
    if not files:
        return False

    num_image_files = len([f for f in files if f.stem != "mask"])
    channels = num_image_files * 3 + (1 if any(f.stem == "mask" for f in files) else 0)
    height = cfg.num_tiles * cfg.tile_size
    width = cfg.num_tiles * cfg.tile_size

    data = np.zeros((height, width, channels), dtype=np.uint8)

    index = 0
    for file_name in files:
        image = read_image(file_name)

        if 3 == len(image.shape):
            data[..., index: index + 3] = image
            index += 3
        else:
            data[..., -1] = image

    filename = f"{str(task_index).zfill(len(str(max_len - 1)))}.h5"
    out_path = dataset_path / filename

    with h5py.File(out_path, "w") as h5f:
        h5f.create_dataset("data", data=data, compression=None, chunks=True)

        h5f.attrs["zoom"] = cfg.zoom
        h5f.attrs["x"] = x
        h5f.attrs["y"] = y

        for class_name, pixel_val in cfg.classes:
            h5f.attrs[class_name] = pixel_val

    return True


def main(cfg: DictConfig):
    sat_dir = Path(cfg.satellite_dir)
    logger.info("Collecting files from %s", sat_dir)
    files = get_files(sat_dir)
    dataset_dir = make_dir(cfg.dataset_dir, delete_if_exist=True)

    tiles_dict: dict[tuple[int, int], list[Path]] = defaultdict(list)

    for f in files:
        parent = f.parent
        grand = parent.parent
        x = int(grand.name)
        y = int(parent.name)
        tiles_dict[(x, y)].append(f)

    items = list(tiles_dict.items())
    total = len(items)
    logger.info("Found %d tiles to process", total)
    if total == 0:
        return

    max_workers = int(cfg.max_workers) if getattr(cfg, "max_workers", None) else min(8, os.cpu_count() or 1)
    logger.info("Using max_workers=%d", max_workers)

    futures = []

    with ThreadPoolExecutor(max_workers) as executor:
        for idx, item in enumerate(items):
            futures.append(
                executor.submit(process_layer, item, cfg, dataset_dir, idx, total)
            )
            logger.info(f"Process item: {idx} / {total}")

    num_processed_items = sum([f.result() for f in futures])

    if num_processed_items != total:
        logger.error(f"Total {total} not equal to processed  {num_processed_items}")


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg.prepare_dataset
    main(cfg)
