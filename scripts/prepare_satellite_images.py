import os
import random
import shutil
from pathlib import Path

import rootutils
from PIL import Image
from joblib import Parallel, delayed, parallel_backend, wrap_non_picklable_objects
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.common import make_dir


def get_tiles(cfg: DictConfig) -> list:
    """Генерирует список тайлов для обработки.

    Args:
        cfg: Конфигурация с параметрами обработки изображений.

    Returns:
        Список тайлов в формате [file, x, y, crop_size].
    """
    crop_size = cfg.crop_size
    tiles = []

    for file in Path(cfg.satellite_dir).glob("**/*.jpg"):
        with Image.open(file) as fp:
            width, height = fp.size

        for y in range(0, height - crop_size, crop_size):
            for x in range(0, width - crop_size, crop_size):
                tiles.append([file, x, y, crop_size])

    return tiles


@delayed
@wrap_non_picklable_objects
def process(item: int, tile: list, data_dir: Path) -> None:
    """Вырезает тайл из изображения и сохраняет его в указанную директорию.

    Args:
        item: Порядковый номер тайла, используется для формирования имени файла.
        tile: Список [file, x, y, crop_size] — путь к исходному изображению,
              координаты левого верхнего угла и размер тайла.
        data_dir: Директория, в которую будет сохранён вырезанный тайл.
    """
    file, x, y, crop_size = tile

    with Image.open(file) as fp:
        area = (x, y, x + crop_size, y + crop_size)
        crop = fp.crop(area)

    crop.save(data_dir / f"{str(item).zfill(6)}.jpg")


# функция для перемещения файлов из временной папки в папку train или test
def move_files(files: list[Path], data_dir: Path) -> None:
    out_dir = make_dir(data_dir / str(0).zfill(3), delete_if_exist=False)
    for n, file in enumerate(files):
        if n % 1000 == 0:
            out_dir = make_dir(data_dir / str(n // 1000).zfill(3), delete_if_exist=False)
        stem = out_dir / str(n % 1000).zfill(3)
        shutil.move(file, stem.with_suffix(file.suffix))


# функция для разделения файлов на обучающий и тестовый датасеты,
# а также для перемещения файлов в соответствующие папки
def split_and_move_files(files: list[Path], train_dir: Path, test_dir: Path, train_size: float):
    random.shuffle(files)
    num_train_files = int(len(files) * train_size)

    train_files = files[:num_train_files]
    test_files = files[num_train_files:]

    move_files(train_files, train_dir)
    move_files(test_files, test_dir)


def main(cfg: DictConfig) -> None:
    # директория для хранения тайлов, обучающего и тестового датасетов
    satellite_data_dir = Path(cfg.images_dir)

    # временная директория для хранения тайлов, которая будет удалена после перемещения файлов в папки train и test
    images_dir = make_dir(satellite_data_dir / "tiles", delete_if_exist=True)

    # создание директорий для хранения тайлов, обучающего и тестового датасетов
    train_dir = make_dir(satellite_data_dir / "train", delete_if_exist=True)
    test_dir = make_dir(satellite_data_dir / "test", delete_if_exist=True)

    # получение списка тайлов для обработки
    tiles = get_tiles(cfg)

    # обработка тайлов в параллельном режиме и сохранение их во временную директорию
    with parallel_backend("threading"):
        Parallel(n_jobs=os.cpu_count())(
            process(item, tiles[item], images_dir)
            for item in tqdm(range(len(tiles)), desc="tiles", leave=False)
        )

    # получение списка файлов с тайлами из временной директории
    # и разделение их на обучающий и тестовый датасеты,
    # а также перемещение файлов в соответствующие директории
    files = list(images_dir.glob("**/*.jpg"))
    split_and_move_files(files, train_dir, test_dir, cfg.train_size)

    # удаление временной папки с тайлами
    shutil.rmtree(images_dir)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.prepare_satellite_images)
