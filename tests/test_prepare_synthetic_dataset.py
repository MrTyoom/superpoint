import cv2
import numpy as np
import pytest
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.utils import make_dir
from src.prepare_synthetic_dataset import generate_data, draw_lines


@pytest.fixture(scope="package")
def init_tests():
    cfg = OmegaConf.load("params.yaml")["prepare_synthetic_dataset"]
    temp_dir = make_dir("./tmp")
    
    return cfg, temp_dir


def test_draw_lines(capsys, init_tests):
    cfg, temp_dir = init_tests

    image_id = 0

    image_file, points_file = generate_data(
        image_id, 
        temp_dir, 
        draw_lines, 
        cfg.background_size, 
        cfg.image_size, 
        cfg.blur_size, 
        cfg.seed
    )

    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape

    assert (H, W) == cfg.image_size
    
    points = np.load(points_file)
    num_points, dim = points.shape
    
    assert num_points > 0
    assert dim == 2

    x = points[:, 0]
    y = points[:, 1]

    ok = (0 <= x) * (x < W) * (0 <= y) * (y < H)

    assert ok.all()

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 3, (255, 0, 255), -1)

    cv2.imwrite(image_file, image)
    