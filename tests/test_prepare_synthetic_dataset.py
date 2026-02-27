from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import cv2
import numpy as np
import pytest
import rootutils
from _pytest.capture import CaptureFixture
from joblib import Parallel
from omegaconf import DictConfig, OmegaConf

# isort: off
rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from scripts.prepare_synthetic_data import (
    draw_checkerboard,
    draw_cube,
    draw_lines,
    draw_multiple_polygons,
    draw_polygon,
    draw_star,
    draw_stripes,
    generate_data,
)

# isort: on
from src.common import make_dir


@pytest.fixture(scope="package")
def init_tests() -> tuple[DictConfig, Path]:
    cfg = OmegaConf.load("params.yaml")["prepare_synthetic_data"]
    temp_dir = make_dir("./tmp")
    return cfg, temp_dir


DRAW_FUNCTIONS = MappingProxyType(
    {
        "lines": (draw_lines, 0),
        "polygon": (draw_polygon, 1),
        "multiple_polygons": (draw_multiple_polygons, 2),
        "star": (draw_star, 3),
        "checkerboard": (draw_checkerboard, 4),
        "stripes": (draw_stripes, 5),
        "cube": (draw_cube, 6),
    }
)


@pytest.mark.parametrize(
    "draw_func_name,draw_func,image_id",
    [(name, func, idx) for name, (func, idx) in DRAW_FUNCTIONS.items()],
)
def test_draw_function(
    capsys: CaptureFixture[str],
    init_tests: tuple[DictConfig, Path],
    draw_func_name: str,
    draw_func: Callable[..., Any],
    image_id: int,
) -> None:
    cfg, temp_dir = init_tests

    delayed_data = generate_data(image_id, temp_dir, draw_func, cfg.background_size, cfg.image_size, cfg.blur_size)
    image_file, points_file = Parallel(n_jobs=1)([delayed_data])[0]

    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    assert (height, width) == cfg.image_size

    points = np.load(points_file)
    num_points, dim = points.shape

    assert num_points > 0, f"image_file: {image_file}"
    assert dim == 2

    x_cord = points[:, 0]
    y_cord = points[:, 1]
    ok = (0 <= x_cord) * (x_cord < width) * (0 <= y_cord) * (y_cord < height)

    assert ok.all(), f"Points outside bounds for {draw_func_name}"

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x_cord, y_cord in points:
        cv2.circle(image_color, (int(x_cord), int(y_cord)), 3, (255, 0, 255), -1)

    cv2.imwrite(image_file, image_color)
