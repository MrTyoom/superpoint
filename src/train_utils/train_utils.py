import cv2
import numpy as np

from src.types import Array, Tensor


def normPts(pts, shape):
    """
    normalize pts to [-1, 1]
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts / shape * 2 - 1
    return pts


def to_numpy(tensor: Tensor) -> Array:
    return tensor.detach().cpu().numpy()


def save_img(img, filename):
    img = img.astype(np.uint8)
    cv2.imwrite(filename, img)


def get_pts_from_heatmap(heatmap: Array, conf_thresh: float, nms_dist: int) -> Array:
    border_remove = 4

    height, width = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.

    if len(xs) == 0:
        return np.zeros((3, 0))

    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]

    pts, _ = nms_fast(pts, height, width, dist_thresh=nms_dist)  # Apply NMS.

    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.

    # Remove points along border.
    bord = border_remove

    width_bord = pts[0, :] >= (width - bord)
    height_bord = pts[1, :] >= (height - bord)

    toremoveW = np.logical_or(pts[0, :] < bord, width_bord)
    toremoveH = np.logical_or(pts[1, :] < bord, height_bord)
    toremove = np.logical_or(toremoveW, toremoveH)

    pts = pts[:, ~toremove]

    return pts


def nms_fast(in_corners: Array, height: int, width: int, dist_thresh: int):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((height, width)).astype(int)  # Track NMS data.
    inds = np.zeros((height, width)).astype(int)  # Store indices of points.

    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.

    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)

    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)

    # Initialize the grid.
    for idx, rc in enumerate(rcorners.T):
        grid[rcorners[1, idx], rcorners[0, idx]] = 1
        inds[rcorners[1, idx], rcorners[0, idx]] = idx

    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode="constant")

    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0

    for idx, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)

        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            x_start = pt[1] - pad
            x_end = pt[1] + pad + 1
            y_start = pt[0] - pad
            y_end = pt[0] + pad + 1
            grid[x_start:x_end, y_start:y_end] = 0
            grid[pt[1], pt[0]] = -1
            count += 1

    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    out_vals = out[-1, :]
    inds2 = np.argsort(-out_vals)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]

    return out, out_inds


def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, "out_num_points = %d must be positive int!" % out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice
