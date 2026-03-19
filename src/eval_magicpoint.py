import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.common import make_dir
from src.logger import LOG
from src.models.superpoint import SuperPoint

# reference model:
# https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
# weights can be downloded from:
# https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth
from src.models.superpoint_reference import SuperPointReference


MAX_PIXEL = 255


class DataSet(Dataset):
    def __init__(self, data_dir):
        super().__init__()

        files = [fp for fp in Path(data_dir).glob("**/*.png") if fp.with_suffix(".npy").exists()]

        self.image_files = files
        self.annot_files = [fp.with_suffix(".npy") for fp in files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)
        points = np.load(self.annot_files[index])

        image = torch.from_numpy(image)
        points = torch.from_numpy(points)

        height, width = image.shape[:2]
        quan_pnts = points.round().long()

        label = torch.zeros((height, width))
        label[quan_pnts[:, 1], quan_pnts[:, 0]] = 1

        return image, label


class DepthToSpace(torch.nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        batch_size, d_height, d_width, d_depth = output.size()

        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)

        tiles = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        tiles = tiles.split(self.block_size, 3)

        stack = [ts.reshape(batch_size, d_height, s_width, s_depth) for ts in tiles]

        output = torch.stack(stack, 0)
        output = output.transpose(0, 1)
        output = output.permute(0, 2, 1, 3, 4)
        output = output.reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)

        return output


def nms_fast(in_corners, height, width, dist_thresh):
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
    for it, rc in enumerate(rcorners.T):
        grid[rcorners[1, it], rcorners[0, it]] = 1
        inds[rcorners[1, it], rcorners[0, it]] = it

    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode="constant")

    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0

    for rc in rcorners.T:
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)

        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            y1 = pt[1] - pad
            y2 = pt[1] + pad + 1
            x1 = pt[0] - pad
            x2 = pt[0] + pad + 1
            grid[y1:y2, x1:x2] = 0
            grid[pt[1], pt[0]] = -1
            count += 1

    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad

    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    inds2 = np.argsort(-out[-1, :])
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]

    return out, out_inds


def get_points(heatmap, conf_thresh=0.015, nms_dist=4):
    height, width = heatmap.shape
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.

    if len(xs) == 0:
        return np.zeros((0,)), np.zeros((0,))

    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.

    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]

    pts, _ = nms_fast(pts, height, width, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])

    pts = pts[:, inds[::-1]]  # Sort by confidence.

    # Remove points along border.
    bord = 4

    ok1 = pts[0, :] < bord
    ok2 = pts[0, :] >= (width - bord)
    toremoveW = np.logical_or(ok1, ok2)

    ok1 = pts[1, :] < bord
    ok2 = pts[1, :] >= (height - bord)
    toremoveH = np.logical_or(ok1, ok2)

    toremove = np.logical_or(toremoveW, toremoveH)

    pts = pts[:, ~toremove]

    xs = pts[0, :].astype(int)
    ys = pts[1, :].astype(int)

    return xs, ys


def load_reference_model(model_file):
    model = SuperPointReference()

    state_dict = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model


def load_trained_model(model_file):
    model = SuperPoint()

    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    state_dict = OrderedDict((key.replace("_net.", ""), weights) for key, weights in checkpoint["state_dict"].items())
    model.load_state_dict(state_dict, strict=True)

    return model


def load_model(model_file):
    ext = Path(model_file).suffix

    if ".ckpt" == ext:
        model = load_trained_model(model_file)
    elif ".pth" == ext:
        model = load_reference_model(model_file)
    else:
        raise NotImplementedError

    return model


def get_heatmap(semi):
    prob = torch.nn.functional.softmax(semi, dim=1)
    heatmap = DepthToSpace(8)(prob[:, :-1, :, :])
    return heatmap.squeeze(1).cpu()


def calc_metrics(x_coords, y_coords, hmap, labl):
    pred = np.zeros_like(hmap, dtype=np.uint8)
    targ = np.zeros_like(labl, dtype=np.uint8)

    for xi, yi in zip(x_coords, y_coords):
        cv2.circle(pred, (xi, yi), 2, (1,), -1)

    y_coords, x_coords = np.where(labl > 0)

    for xj, yj in zip(x_coords, y_coords):
        cv2.circle(targ, (xj, yj), 2, (1,), -1)

    sum_ = (pred * targ).sum()

    EPS = 0.00001
    precision = sum_ / (pred.sum() + EPS)
    recall = sum_ / (targ.sum() + EPS)

    return precision, recall, pred, targ


@torch.inference_mode()
def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    loader = DataLoader(
        DataSet(cfg.data_dir),
        batch_size=cfg.batch_size,
        num_workers=os.cpu_count(),
        shuffle=False,
        drop_last=False,
        pin_memory=is_cuda,
    )

    model = load_model(cfg.checkpoint)
    model = model.eval().to(device)

    output_dir = None if cfg.output_dir is None else make_dir(cfg.output_dir)
    count = 0

    precisions = []
    recalls = []

    for images, labels in tqdm(loader, total=len(loader), leave=False):
        batch = images.unsqueeze(1).float().to(device, non_blocking=is_cuda) / MAX_PIXEL

        semi, _ = model(batch)
        heatmap = get_heatmap(semi)

        for hmap, imag, labl in zip(heatmap.numpy(), images.numpy(), labels.numpy()):
            x_coords, y_coords = get_points(hmap)
            precision, recall, pred, targ = calc_metrics(x_coords, y_coords, hmap, labl)

            precisions.append(precision)
            recalls.append(recall)

            if output_dir is not None:
                out_imag = cv2.cvtColor(imag, cv2.COLOR_GRAY2BGR)

                for xi, yi in zip(x_coords, y_coords):
                    cv2.circle(out_imag, (xi, yi), 3, (255, 0, 255), -1)

                out_file = (output_dir / str(count).zfill(6)).with_suffix(".jpg")
                cv2.imwrite(out_file, out_imag)

                out_imag = np.zeros_like(out_imag)
                out_imag[:, :, 1] = MAX_PIXEL * targ
                out_imag[:, :, 2] = MAX_PIXEL * pred
                cv2.imwrite(out_file.with_suffix(".png"), out_imag)

                count += 1

    mean_precision = torch.tensor(precisions).mean().item()
    mean_recall = torch.tensor(recalls).mean().item()

    LOG.info(f"Model: {cfg.checkpoint}")
    LOG.info(f"Precision: {mean_precision: .4f}")
    LOG.info(f"Recall: {mean_recall: .4f}")


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.eval_magicpoint)
