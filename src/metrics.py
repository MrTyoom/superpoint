import numpy as np
import torch

from src.train_utils.d2s import DepthToSpace
from src.train_utils.train_utils import get_pts_from_heatmap
from src.types import Tensor


def get_heatmap(semi: Tensor) -> Tensor:
    semi_soft = torch.nn.functional.softmax(semi, dim=1)
    nodust = semi_soft[:, :-1, :, :]

    depth_to_space = DepthToSpace(8)
    heatmap = depth_to_space(nodust)

    return heatmap


def get_precision_recall(pred: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    OFFSET = 10**-6

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


def batch_precision_recall(
    heatmap: Tensor, labels: Tensor, masks: Tensor, detection_threshold: float, nms_dist: int
) -> tuple[float, float]:

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
