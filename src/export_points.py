import random
from pathlib import Path

import cv2
import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.common import make_dir
from src.eval_magicpoint import get_pts_from_heatmap, load_trained_model
from src.metrics import get_heatmap
from src.transform import Augmentation


MAX_PIXEL_VALUE = 255


def save_debug_images(heatmap, image, key_points, debug_dir: Path, file_name: Path):
    hmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    hmap = (MAX_PIXEL_VALUE * hmap).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)

    x_coords = key_points[:, 0].astype(int)
    y_coords = key_points[:, 1].astype(int)

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for xy in zip(x_coords, y_coords):
        cv2.circle(hmap, xy, 1, (0, 0, 255), -1)
        cv2.circle(img, xy, 1, (255, 0, 255), -1)

    out_dir = make_dir(debug_dir / file_name.parent.name, delete_if_exist=False)

    cv2.imwrite(out_dir / file_name.name, img)
    cv2.imwrite(out_dir / Path(file_name.name).with_suffix(".png"), hmap)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(cfg):
    seed_everything(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # загузка предобученной модели MagicPoint на синтетическом датасете
    model = load_trained_model(cfg.magic_point_model)
    model = model.eval()
    model = model.to(device)

    debug_dir = None if cfg.debug_dir is None else make_dir(cfg.debug_dir, delete_if_exist=True)

    # список исходных спутниковых изображений
    files = sorted(Path(cfg.images_dir).glob("**/*.jpg"))

    augmentation = Augmentation(cfg.augmentation)

    for image_file in tqdm(files, total=len(files), leave=False):
        # чтение изображения и сглаживание для уменьшения шума
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        blur_image = cv2.blur(image, (cfg.blur_size, cfg.blur_size))

        # нормализация изображения и создание батча - копий одного изображения (B x C x H x W)
        batch = torch.tensor(blur_image, dtype=torch.float, device=device).unsqueeze(0)
        batch = torch.repeat_interleave(batch / MAX_PIXEL_VALUE, cfg.batch_size, dim=0).unsqueeze(1)

        # геометрическая и цветовая аугментация батча
        batch = augmentation(batch)

        # инференс сети
        with torch.inference_mode():
            semi, _ = model(batch)

        # постобработка для получения тепловых карт ключевых точек
        heatmaps = get_heatmap(semi)
        # обратное геометрическое преобразование тепловых карт
        heatmaps = augmentation.warp(heatmaps)
        # усреднение тепловых карт (homography adaptation)
        heatmap = torch.mean(heatmaps, dim=0).squeeze()  # H x W
        heatmap = heatmap.cpu().numpy()

        # вычисление ключевых точек
        key_points = get_pts_from_heatmap(heatmap, cfg.detection_threshold, cfg.nms_dist)
        key_points = key_points.transpose()
        # сохранение ключевых точек
        output_file = image_file.with_suffix(".npy")
        np.save(output_file, key_points.astype(np.float32))

        if debug_dir is not None:
            save_debug_images(heatmap, image, key_points, debug_dir, image_file)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.export_points)
