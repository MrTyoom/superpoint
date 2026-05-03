from pathlib import Path

import cv2
import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.common import make_dir
from src.eval_magicpoint import load_model
from src.metrics import get_heatmap
from src.train_utils.train_utils import get_pts_from_heatmap
from src.transform import Augmentation


MAX_PIXEL_VALUE = 255.0
THRESHOLD = 0.1


def get_pts(semi, cfg, augmentation=None):
    # постобработка для получения тепловых карт ключевых точек
    heatmaps = get_heatmap(semi)

    if augmentation is not None:
        # обратное геометрическое преобразование тепловых карт
        heatmaps = augmentation.warp(heatmaps)

    # усреднение тепловых карт (homography adaptation)
    src_heatmap = torch.mean(heatmaps, dim=0).squeeze()  # H x W
    src_heatmap = src_heatmap.cpu().numpy()

    src_keypoints = get_pts_from_heatmap(src_heatmap, cfg.detection_threshold, cfg.nms_dist)
    src_keypoints = src_keypoints.transpose()

    return src_keypoints


def update_desc(coarse_desc, pts, cell_size=8):
    dense_desc = torch.nn.functional.interpolate(coarse_desc, scale_factor=(cell_size, cell_size), mode="bilinear")
    dense_desc = norm_desc(dense_desc)

    dense_desc_cpu = dense_desc.cpu().detach().numpy()

    x = pts[:, 0].astype(int)
    y = pts[:, 1].astype(int)

    pts_desc = dense_desc_cpu[0, :, y, x].transpose()
    return pts_desc


def norm_desc(desc):
    dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
    return desc


def cross_check_matching(D, threshold):
    best_from_des2_to_des1 = np.argmin(D, axis=1)
    best_from_des1_to_des2 = np.argmin(D, axis=0)
    min_distances_des2_to_des1 = np.min(D, axis=1)
    min_distances_des1_to_des2 = np.min(D, axis=0)

    des2_indices = np.arange(D.shape[0])
    cross_check_mask = (
        (best_from_des1_to_des2[best_from_des2_to_des1] == des2_indices)
        & (min_distances_des2_to_des1 < threshold)
        & (min_distances_des1_to_des2[best_from_des2_to_des1] < threshold)
    )

    valid_des2_indices = des2_indices[cross_check_mask]
    valid_des1_indices = best_from_des2_to_des1[cross_check_mask]

    matches = list(zip(valid_des1_indices, valid_des2_indices))
    return matches


def matches_vizualize(batches, coords, matches):
    src_batch, dst_batch = batches
    src_x_coords, src_y_coords, dst_x_coords, dst_y_coords = coords

    src_img = (src_batch[0, 0].cpu().numpy() * 255).astype(np.uint8)
    dst_img = (dst_batch[0, 0].cpu().numpy() * 255).astype(np.uint8)

    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_GRAY2BGR)

    output_image = np.concatenate((src_img, dst_img), axis=1)
    _, _, H, W = dst_batch.shape

    for idx1, idx2 in matches:
        x = src_x_coords[idx1]
        y = src_y_coords[idx1]
        cv2.circle(output_image, (int(x), int(y)), 4, (255, 0, 255), -1)

    for idx1, idx2 in matches:
        x = dst_x_coords[idx2]
        y = dst_y_coords[idx2]
        colour = (255, 0, 255)
        cv2.circle(output_image, (int(x) + W, int(y)), 4, colour, -1)

    for idx1, idx2 in matches:
        x1 = src_x_coords[idx1]
        y1 = src_y_coords[idx1]
        x2 = dst_x_coords[idx2]
        y2 = dst_y_coords[idx2]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.line(output_image, (x1, y1), (x2 + W, y2), (0, 255, 0), 1)

    return output_image


def filter_valid_points(pts, mask):
    x = pts[:, 0].astype(int)
    y = pts[:, 1].astype(int)
    valid = mask[y, x] > 0
    return pts[valid]


@torch.inference_mode()
def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    augmentation = Augmentation(cfg.augmentation)
    files = sorted(Path(cfg.data_dir).glob("**/*.jpg"))

    model = load_model(cfg.checkpoint)
    model = model.eval().to(device)

    output_dir = make_dir(cfg.output_dir, delete_if_exist=True)
    count = 0

    for image_file in tqdm(files, total=len(files), leave=False):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        blur_image = cv2.blur(image, (cfg.blur_size, cfg.blur_size))

        # нормализация изображения и создание батча - копий одного изображения (B x C x H x W)
        src_batch = torch.tensor(blur_image, dtype=torch.float, device=device).unsqueeze(0)
        src_batch = torch.repeat_interleave(src_batch / MAX_PIXEL_VALUE, cfg.batch_size, dim=0).unsqueeze(1)

        dst_batch = augmentation(src_batch)
        valid_mask = (dst_batch[0, 0] > THRESHOLD).cpu().numpy()

        src_semi, src_desc = model(src_batch)
        dst_semi, dst_desc = model(dst_batch)

        src_keypoints = get_pts(src_semi, cfg)
        dst_keypoints = get_pts(dst_semi, cfg, augmentation=augmentation)

        dst_keypoints = filter_valid_points(dst_keypoints, valid_mask)

        src_x_coords = src_keypoints[:, 0].astype(int)
        src_y_coords = src_keypoints[:, 1].astype(int)

        dst_x_coords = dst_keypoints[:, 0].astype(int)
        dst_y_coords = dst_keypoints[:, 1].astype(int)

        src_desc = update_desc(src_desc, src_keypoints)
        dst_desc = update_desc(dst_desc, dst_keypoints)

        src_desc = src_desc.T
        dst_desc = dst_desc.T

        dots = np.dot(dst_desc, src_desc.T)
        D = 2 * (1 - np.clip(dots, -1, 1))

        matches = cross_check_matching(D, 0.7)

        output_image = matches_vizualize(
            (src_batch, dst_batch), (src_x_coords, src_y_coords, dst_x_coords, dst_y_coords), matches
        )

        out_file = (output_dir / str(count).zfill(3)).with_suffix(".jpg")
        cv2.imwrite(out_file, output_image)

        count += 1


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.eval_superpoint)
