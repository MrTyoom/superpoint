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
from src.metrics import get_heatmap
from src.models.superpoint import SuperPoint

# reference model:
# https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
# weights can be downloded from:
# https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth
from src.models.superpoint_reference import SuperPointReference
from src.train_utils.train_utils import get_pts_from_heatmap


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

    output_dir = None if cfg.output_dir is None else make_dir(cfg.output_dir, delete_if_exist=True)
    count = 0

    precisions = []
    recalls = []

    for images, labels in tqdm(loader, total=len(loader), leave=False):
        batch = images.unsqueeze(1).float().to(device, non_blocking=is_cuda) / MAX_PIXEL

        semi, _ = model(batch)
        heatmap = get_heatmap(semi).squeeze(1).cpu()

        for hmap, imag, labl in zip(heatmap.numpy(), images.numpy(), labels.numpy()):
            coords = get_pts_from_heatmap(hmap, conf_thresh=cfg.detection_threshold, nms_dist=cfg.nms_dist)
            x_coords = coords[0, :].astype(int)
            y_coords = coords[1, :].astype(int)
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
