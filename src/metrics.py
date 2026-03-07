import torch

from src.train_utils.d2s import DepthToSpace
from src.train_utils.train_utils import getPtsFromHeatmap


def get_heatmap(semi):
    semi_soft = torch.nn.functional.softmax(semi, dim=1)
    nodust = semi_soft[:, :-1, :, :]

    depth_to_space = DepthToSpace(8)
    heatmap = depth_to_space(nodust)
    return heatmap


def precisionRecall(pred, labels):
    offset = 10**-6
    precision = torch.sum(pred * labels) / (torch.sum(pred) + offset)
    recall = torch.sum(pred * labels) / (torch.sum(labels) + offset)
    return {"precision": precision, "recall": recall}


def calculate_precisionRecall(batch, heatmap, lab_mask, detection_threshold, nms_dist):
    labels, masks = lab_mask

    heatmap_np = heatmap[batch, 0]
    heatmap_np = heatmap_np.cpu().numpy()
    pts_nms = getPtsFromHeatmap(heatmap_np, detection_threshold, nms_dist)  # (2, N)

    pred_map = torch.zeros_like(labels[batch, 0]).cpu()
    pts_nms = torch.from_numpy(pts_nms).long()
    if pts_nms.shape[1] > 0:
        pred_map[pts_nms[1, :].long(), pts_nms[0, :].long()] = 1

    gt_map = (labels[batch, 0] * masks[batch, 0]).cpu()

    pr = precisionRecall(pred_map, gt_map)
    return pr
