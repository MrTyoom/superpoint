from pathlib import Path

import cv2
import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from usearch.index import Index


rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.eval_magicpoint import load_model
from src.eval_superpoint import filter_valid_points, get_pts, update_desc
from src.export_points import seed_everything
from src.transform import Augmentation


MAX_PIXEL_VALUE = 255.0
DESCRIPTOR_SIZE = 256
LOWE_RATIO = 0.75
MIN_MATCHES = 4
THRESHOLD = 0.5


def visualise(orig_img, tile, H):
    orig = cv2.imread(str(orig_img), cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(orig_img)

    orig_color = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    _, _, _, crop_size = tile

    pts = np.array(
        [[0, 0], [crop_size, 0], [crop_size, crop_size], [0, crop_size]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    warped_pts = cv2.perspectiveTransform(pts, H)
    cv2.polylines(orig_color, [warped_pts.astype(int)], True, (0, 255, 0), 3)
    cv2.imwrite("res_img.png", orig_color)


def localize_crop(global_pts, global_desc, crop_pts, crop_desc):
    global_desc = np.ascontiguousarray(global_desc, dtype=np.float32)
    crop_desc = np.ascontiguousarray(crop_desc, dtype=np.float32)

    index = Index(ndim=DESCRIPTOR_SIZE, metric="l2sq")
    index.add(np.arange(len(global_desc)), global_desc)

    matches = index.search(crop_desc, 2)

    good = []
    for i, (ks, ds) in enumerate(zip(matches.keys, matches.distances)):
        if len(ks) >= 2 and ds[0] < (LOWE_RATIO**2) * ds[1]:
            good.append((i, ks[0]))

    if len(good) < MIN_MATCHES:
        raise RuntimeError(f"Не достаточно матчей дескрипторов: {len(good)}")

    src_pts = np.array([crop_pts[idx] for idx, _ in good], dtype=np.float32)
    dst_pts = np.array([global_pts[matched_id] for _, matched_id in good], dtype=np.float32)

    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
    if H is None or inliers is None or int(inliers.sum()) < MIN_MATCHES:
        raise RuntimeError("Не удалось построить гомографию")

    return H


def get_tiles_simple(file_name, crop_size=320):
    tiles = []

    with Image.open(file_name) as fp:
        width, height = fp.size

    for y in range(0, height - crop_size + 1, crop_size):
        for x in range(0, width - crop_size + 1, crop_size):
            tiles.append([file_name, x, y, crop_size])

    return tiles


def extract_features(cfg, image, model, device, augmentation=None):
    blur_image = cv2.blur(image, (cfg.blur_size, cfg.blur_size))

    batch = torch.tensor(blur_image, dtype=torch.float, device=device)
    batch = (batch / MAX_PIXEL_VALUE).unsqueeze(0).unsqueeze(0)

    if augmentation is None:
        valid_mask = None
    else:
        geometry_mask = torch.ones_like(batch)
        batch = augmentation(batch)
        valid_mask = torch.grid_sampler(
            geometry_mask,
            augmentation.inv_grid,
            1,
            0,
            align_corners=True,
        )
        valid_mask = (valid_mask[0, 0] > THRESHOLD).cpu().numpy()

    with torch.inference_mode():
        semi, desc = model(batch)

    keypoints = get_pts(semi, cfg)
    if valid_mask is not None:
        keypoints = filter_valid_points(keypoints, valid_mask)

    desc = update_desc(desc, keypoints).T
    norm = np.linalg.norm(desc, axis=1, keepdims=True)
    desc = desc / (norm + 1e-7)

    return keypoints[:, :2].astype(np.float32), desc.astype(np.float32)


def train_samples(cfg, tiles, model, device, augmentation=None, is_global=False):  # noqa: WPS211
    all_pts = []
    all_desc = []

    for tile in tqdm(tiles, total=len(tiles), leave=False):
        file_name, x, y, crop_size = tile

        orig_img = cv2.imread(str(file_name), cv2.IMREAD_GRAYSCALE)

        image = orig_img[y : y + crop_size, x : x + crop_size]

        if len(tiles) == 1:
            cv2.imwrite("cur_tile.png", image)

        src_keypoints, src_desc = extract_features(cfg, image, model, device, augmentation)

        if is_global:
            src_keypoints[:, 0] += x
            src_keypoints[:, 1] += y

        all_pts.append(src_keypoints)
        all_desc.append(src_desc)

    if len(tiles) == 1:
        return all_pts[0], all_desc[0]

    all_pts = np.concatenate(all_pts, axis=0)
    all_desc = np.concatenate(all_desc, axis=0)
    return all_pts, all_desc


def main(cfg):
    seed_everything(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cur_path = Path(cfg.data_dir) / "forest"
    augmentation = Augmentation(cfg.augmentation)

    images = sorted(cur_path.glob("*.jpg"))
    rand_domen = torch.randint(0, len(images), (1,)).item()
    img = images[rand_domen]

    tiles = get_tiles_simple(img)

    rand_idx = torch.randint(0, len(tiles), (1,)).item()
    rand_tile = tiles[rand_idx + 5]

    model = load_model(cfg.checkpoint)
    model = model.eval().to(device)

    global_pts, global_desc = train_samples(cfg, tiles, model, device, is_global=True)
    crop_tile_pts, crop_tile_desc = train_samples(
        cfg,
        [rand_tile],
        model,
        device,
        augmentation=augmentation,
        is_global=False,
    )

    H = localize_crop(global_pts, global_desc, crop_tile_pts, crop_tile_desc)
    visualise(img, rand_tile, H)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.localization_tests)
