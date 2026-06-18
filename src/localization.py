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
from src.metrics import compute_center_error, compute_corner_error, compute_iou_pr
from src.transform import Augmentation


MAX_PIXEL_VALUE = 255.0
DESCRIPTOR_SIZE = 256
LOWE_RATIO = 0.75
MIN_MATCHES = 4
THRESHOLD = 0.5
TOP_K = 250


def pixel_transform(h, w):
    w_d = 1e-14 if w == 1 else w - 1.0
    h_d = 1e-14 if h == 1 else h - 1.0
    return np.array(
        [
            [2 / w_d, 0, -1],
            [0, 2 / h_d, -1],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def denormalize_homography(H_norm, height, width):
    src_pix = pixel_transform(height, width)
    dst_pix = pixel_transform(height, width)

    return np.linalg.inv(dst_pix) @ H_norm @ src_pix


def evaluate_localization(H_pred, H_gt, tiles, image_shape):
    center_error = compute_center_error(H_pred, H_gt, tiles)
    corner_error = compute_corner_error(H_pred, H_gt, tiles)
    iou, precision, recall, f1 = compute_iou_pr(H_pred, H_gt, tiles, image_shape)

    return {
        "center_error_px": center_error,
        "corner_error_px": corner_error,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def visualise_set(orig_img_path, tiles, H_pred, H_gt):
    orig = cv2.imread(str(orig_img_path))
    if len(orig.shape) == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

    base_x, base_y = tiles[0][1], tiles[0][2]

    for tile in tiles:
        _, x, y, size = tile
        local_x = x - base_x
        local_y = y - base_y

        pts = np.array(
            [
                [local_x, local_y],
                [local_x + size, local_y],
                [local_x + size, local_y + size],
                [local_x, local_y + size],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        # Предсказание — зелёным
        pred_pts = cv2.perspectiveTransform(pts, H_pred)
        cv2.polylines(orig, [pred_pts.astype(np.int32)], True, (0, 255, 0), 2)

        # GT — красным
        gt_pts = cv2.perspectiveTransform(pts, H_gt)
        cv2.polylines(orig, [gt_pts.astype(np.int32)], True, (0, 0, 255), 2)

    # Легенда
    cv2.putText(orig, "Pred", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(orig, "GT", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imwrite("res_set_localization.png", orig)


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

    print(f"Good: {good}")
    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 1.0)
    print(f"Inliers: {inliers}")
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

        img_to_save = (batch[0, 0].detach().cpu().numpy() * 255).astype("uint8")
        cv2.imwrite("warped_crop4.jpg", img_to_save)

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

    if keypoints.shape[0] <= TOP_K:
        return keypoints.astype(np.float32), desc.astype(np.float32)

    idx = np.argsort(keypoints[:, 2])[::-1][:TOP_K]

    top_k_keypoints = keypoints[idx]
    top_k_desc = desc[idx]

    return top_k_keypoints.astype(np.float32), top_k_desc.astype(np.float32)


def train_samples(cfg, tiles, model, device, augmentation=None, is_global=False):
    all_pts = []
    all_desc = []

    base_x, base_y = tiles[0][1], tiles[0][2]

    if augmentation is not None:
        strips = []
        for _, x, y, size in tiles:
            orig_img = cv2.imread(str(tiles[0][0]), cv2.IMREAD_GRAYSCALE)
            strips.append(orig_img[y : y + size, x : x + size])

        wide_image = np.concatenate(strips, axis=1)
        return extract_features(cfg, wide_image, model, device, augmentation)

    for tile in tqdm(tiles, total=len(tiles), leave=False):
        file_name, x, y, crop_size = tile

        orig_img = cv2.imread(str(file_name), cv2.IMREAD_GRAYSCALE)
        image = orig_img[y : y + crop_size, x : x + crop_size]
        src_keypoints, src_desc = extract_features(cfg, image, model, device)

        if is_global:
            src_keypoints[:, 0] += x
            src_keypoints[:, 1] += y
        else:
            src_keypoints[:, 0] += x - base_x
            src_keypoints[:, 1] += y - base_y

        all_pts.append(src_keypoints)
        all_desc.append(src_desc)

    all_pts = np.concatenate(all_pts, axis=0)
    all_desc = np.concatenate(all_desc, axis=0)
    return all_pts, all_desc


def main(cfg):
    seed_everything(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cur_path = Path(cfg.data_dir) / "city"
    augmentation = Augmentation(cfg.augmentation)

    images = sorted(cur_path.glob("*.jpg"))
    rand_domen = torch.randint(0, len(images), (1,)).item()
    img = images[rand_domen]

    orig_img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)

    tiles = get_tiles_simple(img)

    rand_idx = torch.randint(0, len(tiles) - 2, (1,)).item()
    rand_tile1 = tiles[rand_idx]
    rand_tile2 = tiles[rand_idx + 1]
    rand_tile3 = tiles[rand_idx + 2]

    model = load_model(cfg.checkpoint)
    model = model.eval().to(device)

    all_rand_tiles = [rand_tile1, rand_tile2, rand_tile3]
    # all_rand_tiles = [rand_tile1]

    global_pts, global_desc = train_samples(cfg, tiles, model, device, is_global=True)
    crop_tile_pts, crop_tile_desc = train_samples(cfg, all_rand_tiles, model, device, augmentation)

    H_aug_norm = augmentation.homography[0].cpu().numpy()

    wide_w = all_rand_tiles[0][3] * len(all_rand_tiles)
    wide_h = all_rand_tiles[0][3]

    H_aug_px = denormalize_homography(H_aug_norm, wide_h, wide_w)

    base_x, base_y = all_rand_tiles[0][1], all_rand_tiles[0][2]
    T = np.array([[1, 0, base_x], [0, 1, base_y], [0, 0, 1]], dtype=np.float32)
    H_gt = T @ np.linalg.inv(H_aug_px)

    H_pred = localize_crop(global_pts, global_desc, crop_tile_pts, crop_tile_desc)

    metrics = evaluate_localization(H_pred, H_gt, all_rand_tiles, orig_img.shape)
    visualise_set(img, all_rand_tiles, H_pred, H_gt)

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")

    main(cfg.localization_tests)
