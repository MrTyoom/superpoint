import cv2
import numpy as np
import torch

from src.train_utils.d2s import DepthToSpace
from src.train_utils.train_utils import get_pts_from_heatmap
from src.types import Tensor


OFFSET = 10**-6


def get_heatmap(semi: Tensor) -> Tensor:
    semi_soft = torch.nn.functional.softmax(semi, dim=1)
    nodust = semi_soft[:, :-1, :, :]

    depth_to_space = DepthToSpace(8)
    heatmap = depth_to_space(nodust)

    return heatmap


def get_precision_recall(pred: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    precision = torch.sum(pred * labels) / (torch.sum(pred) + OFFSET)
    recall = torch.sum(pred * labels) / (torch.sum(labels) + OFFSET)

    return precision, recall


def calculate_precision_recall(
    heatmap: Tensor, labels: Tensor, masks: Tensor, detection_threshold: float, nms_dist: int
) -> tuple[Tensor, Tensor]:

    heatmap_np = heatmap.cpu().numpy()
    pts_nms = get_pts_from_heatmap(heatmap_np, detection_threshold, nms_dist)

    pred_map = torch.zeros_like(labels).cpu()
    pts_nms = torch.from_numpy(pts_nms).long()

    if pts_nms.shape[1] > 0:
        pred_map[pts_nms[1, :], pts_nms[0, :]] = 1

    gt_map = (labels * masks).cpu()

    precision, recall = get_precision_recall(pred_map, gt_map)

    return precision, recall


def metric_calculation(
    semi: Tensor, labels: Tensor, masks: Tensor | None, detection_threshold: float, nms_dist: int
) -> tuple[float, float]:

    if masks is None:
        masks = torch.ones_like(labels)

    heatmap = get_heatmap(semi)
    batch_size = heatmap.shape[0]

    batch_precision = np.zeros(batch_size)
    batch_recall = np.zeros(batch_size)

    for it in range(batch_size):
        precision, recall = calculate_precision_recall(
            heatmap[it, 0], labels[it, 0], masks[it, 0], detection_threshold, nms_dist
        )

        batch_precision[it] = precision
        batch_recall[it] = recall

    return batch_precision.mean(), batch_recall.mean()


def get_tile_corners(tiles):
    base_x, base_y = tiles[0][1], tiles[0][2]
    corners = []
    for _, x, y, size in tiles:
        lx, ly = x - base_x, y - base_y
        corners.append(
            [
                [lx, ly],
                [lx + size, ly],
                [lx + size, ly + size],
                [lx, ly + size],
            ]
        )
    return np.array(corners, dtype=np.float32)  # (N, 4, 2)


def warp_corners(corners, H):
    n = corners.shape[0]
    pts = corners.reshape(-1, 1, 2)  # (N*4, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(n, 4, 2)  # (N, 4, 2)


def compute_center_error(H_pred, H_gt, tiles):
    corners = get_tile_corners(tiles)

    pred_corners = warp_corners(corners, H_pred)  # (N, 4, 2)
    gt_corners = warp_corners(corners, H_gt)

    pred_center = pred_corners.reshape(-1, 2).mean(axis=0)  # (2,)
    gt_center = gt_corners.reshape(-1, 2).mean(axis=0)

    return float(np.linalg.norm(pred_center - gt_center))


def compute_corner_error(H_pred, H_gt, tiles):
    """Средняя ошибка по всем углам всех тайлов."""
    corners = get_tile_corners(tiles)

    pred_corners = warp_corners(corners, H_pred)  # (N, 4, 2)
    gt_corners = warp_corners(corners, H_gt)

    errors = np.linalg.norm(pred_corners - gt_corners, axis=2)  # (N, 4)
    return float(errors.mean())


def make_mask(H, tiles, image_shape):
    h, w = image_shape[:2]
    corners = get_tile_corners(tiles)

    mask = np.zeros((h, w), dtype=np.uint8)
    warped = warp_corners(corners, H)  # (N, 4, 2)
    for tile_corners in warped:
        cv2.fillPoly(mask, [tile_corners.astype(np.int32)], 1)
    return mask


def compute_iou_pr(H_pred, H_gt, tiles, image_shape):
    pred_mask = make_mask(H_pred, tiles, image_shape)
    gt_mask = make_mask(H_gt, tiles, image_shape)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_area = pred_mask.sum()
    gt_area = gt_mask.sum()

    iou = intersection / union if union > 0 else 0
    precision = intersection / pred_area if pred_area > 0 else 0
    recall = intersection / gt_area if gt_area > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return float(iou), float(precision), float(recall), float(f1)
