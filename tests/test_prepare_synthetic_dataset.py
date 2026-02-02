import cv2
import numpy as np
import pytest
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.prepare_synthetic_dataset import (
    draw_checkerboard,
    draw_cube,
    draw_lines,
    draw_multiple_polygons,
    draw_polygon,
    draw_star,
    draw_stripes,
    generate_data,
)
from src.utils.utils import make_dir


@pytest.fixture(scope="package")
def init_tests():
    cfg = OmegaConf.load("params.yaml")["prepare_synthetic_dataset"]
    temp_dir = make_dir("./tmp")
    return cfg, temp_dir


DRAW_FUNCTIONS = {
    "lines": (draw_lines, 0),
    "polygon": (draw_polygon, 1),
    "multiple_polygons": (draw_multiple_polygons, 2),
    "star": (draw_star, 3),
    "checkerboard": (draw_checkerboard, 4),
    "stripes": (draw_stripes, 5),
    "cube": (draw_cube, 6),
}


@pytest.mark.parametrize(
    "draw_func_name,draw_func,image_id",
    [(name, func, idx) for name, (func, idx) in DRAW_FUNCTIONS.items()],
)
def test_draw_function(capsys, init_tests, draw_func_name, draw_func, image_id):
    cfg, temp_dir = init_tests

    image_file, points_file = generate_data(
        image_id,
        temp_dir,
        draw_func,
        cfg.background_size,
        cfg.image_size,
        cfg.blur_size,
        cfg.seed,
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

    assert ok.all(), f"Points outside bounds for {draw_func_name}"

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(image_color, (int(x), int(y)), 3, (255, 0, 255), -1)

    cv2.imwrite(image_file, image_color)
