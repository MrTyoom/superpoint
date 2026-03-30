from pathlib import Path

import cv2
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


rootutils.setup_root(__file__, "src", pythonpath=True)

from src.common import make_dir
from src.eval_magicpoint import load_trained_model
from src.homography.apply import inv_warp_image
from src.homography.homography_utils import compute_valid_mask, homography_adaptation, sample_homography
from src.loss import soft_argmax_points
from src.metrics import get_heatmap
from src.train_utils.train_utils import get_pts_from_heatmap, save_img


MAX_PIXEL = 255


class DataSet(Dataset):
    def __init__(self, data_dir: Path, aug_cfg: DictConfig, num_homographies: int = 50):
        super().__init__()

        files = [fp for fp in Path(data_dir).glob("**/*.jpg")]
        self.image_files = files

        self.aug_cfg = aug_cfg
        self.num_homographies = num_homographies

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        src_image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)
        src_tensor = torch.from_numpy(src_image).float() / MAX_PIXEL

        height, width = src_image.shape

        warped_images = []
        inv_homographies = []
        masks = []

        # Indentity homography
        img = torch.from_numpy(src_image).float() / MAX_PIXEL

        warped_images.append(img.unsqueeze(0))
        inv_homographies.append(torch.eye(3))

        mask = torch.ones((1, height, width))
        masks.append(mask)

        for _ in range(self.num_homographies - 1):
            homography, inv_homography = sample_homography(self.aug_cfg, np.array([2, 2]), shift=-1)

            warped = inv_warp_image(img.squeeze(), inv_homography, mode="bilinear").unsqueeze(0)

            mask = compute_valid_mask(
                torch.tensor([height, width]),
                inv_homography=inv_homography,
                erosion_radius=self.aug_cfg.valid_border_margin,
            )

            warped_images.append(warped)
            inv_homographies.append(inv_homography)
            masks.append(mask)

        warped_images = torch.stack(warped_images, dim=0)

        masks = torch.stack(masks, dim=0)

        inv_homographies = torch.stack(inv_homographies, dim=0)

        return src_tensor.unsqueeze(0), warped_images, masks, inv_homographies


def draw_keypoints(img, corners, color=(255, 0, 255), radius=3):
    """
    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    """

    img_colored = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x_cord, y_cord in corners:
        cv2.circle(img_colored, (int(x_cord.item()), int(y_cord.item())), radius, color, -1)
    return img_colored


@torch.no_grad()
def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    save_npy = None if cfg.output_dir is None else make_dir(Path(cfg.output_dir) / "keypoints", delete_if_exist=True)
    save_vis = (
        None if cfg.output_dir is None else make_dir(Path(cfg.output_dir) / "visualizations", delete_if_exist=True)
    )

    loader = DataLoader(
        DataSet(cfg.data_dir, cfg.homographic),
        batch_size=1,
        num_workers=cfg.max_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=is_cuda,
    )

    model = load_trained_model(cfg.checkpoint)
    model = model.eval().to(device)

    for idx, (src_img, warped_img, masks, inv_homo) in enumerate(tqdm(loader)):
        warped_img = warped_img.squeeze(0)
        batch = warped_img.float().to(device, non_blocking=is_cuda)

        semi, _ = model(batch)
        heatmap = get_heatmap(semi).cpu()

        outputs = homography_adaptation(heatmap, inv_homo.squeeze(0), masks.squeeze(0), device="cpu")
        pts = get_pts_from_heatmap(
            outputs.detach().cpu().squeeze() * MAX_PIXEL, conf_thresh=cfg.detection_threshold, nms_dist=cfg.nms_dist
        )

        # subpixel enable
        pts = soft_argmax_points([pts], outputs)
        pts = pts[0]

        pts = pts.transpose()

        if pts.shape[0] > cfg.top_k:
            pts = pts[: cfg.top_k, :]

        file_idx = idx + 1
        npy_path = save_npy / f"{file_idx}.npy"
        np.save(npy_path, pts)

        pts_src = pts[:, :2]
        img_pts = draw_keypoints(src_img.numpy().squeeze() * MAX_PIXEL, pts_src.squeeze())
        vis_path = save_vis / f"{file_idx}.png"
        save_img(img_pts, str(vis_path))


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.homographic_adaptation)
