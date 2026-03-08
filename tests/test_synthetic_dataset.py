import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import rootutils
from omegaconf import DictConfig, OmegaConf

from src.common import make_dir
from src.types import Array, Tensor

root = rootutils.setup_root(__file__, indicator="src", pythonpath=True)
EPS = 1e-8
GRAY_CMAP = "gray"

from src.synthetic_loader import SyntheticDataset


@pytest.fixture(scope="module")
def config() -> DictConfig:
    """Загружает конфигурацию для тестов"""
    cfg = OmegaConf.load(root / "params.yaml")
    return cfg.train_magicpoint


@pytest.fixture(scope="module")
def synthetic_dataset(config: DictConfig) -> SyntheticDataset:
    """Создает датасет для тестов"""
    cfg = config.copy()

    data_path = root / cfg["data_dir"]
    files = list(Path(data_path).glob("**/*.png"))
    files = [cur_file for cur_file in files if cur_file.with_suffix(".npy").exists()]
    return SyntheticDataset(files, cfg["augmentation"])


def test_dataset_item_shape(synthetic_dataset: SyntheticDataset) -> None:
    """Тестирует форму элементов датасета (3 элементов)"""
    sample = synthetic_dataset[0]
    assert len(sample) == 3, f"Expected 3 items, got {len(sample)}"

    valid_mask = sample[1]
    labels_two_dim = sample[2]

    assert all(torch.is_tensor(sample_el) for sample_el in sample)

    labels_two_dim_squeezed = labels_two_dim.squeeze()
    valid_mask_squeezed = valid_mask.squeeze()

    assert labels_two_dim_squeezed.dim() == 2
    assert valid_mask_squeezed.dim() == 2

    height = labels_two_dim_squeezed.shape[0]
    width = labels_two_dim_squeezed.shape[1]

    assert valid_mask_squeezed.shape == (height, width)

    cfg = OmegaConf.load(root / "params.yaml")
    expected_size = cfg.prepare_synthetic_data.image_size

    assert (height, width) == tuple(expected_size) or (width, height) == tuple(expected_size)


def test_dataloader_works(synthetic_dataset: SyntheticDataset) -> None:
    """Тестирует работу DataLoader с датасетом"""
    dataloader = torch.utils.data.DataLoader(
        synthetic_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True
    )

    batch = next(iter(dataloader))
    assert len(batch) == 3, f"Expected 3 items in batch, got {len(batch)}"

    img_batch = batch[0]
    valid_mask_batch = batch[1]
    labels_two_dim_batch = batch[2]

    assert img_batch.shape[0] == 4
    assert labels_two_dim_batch.shape[0] == 4
    assert valid_mask_batch.shape[0] == 4


def test_keypoints_statistics(synthetic_dataset: SyntheticDataset) -> None:
    """Тестирует статистику ключевых точек"""
    sample = synthetic_dataset[0]
    labels_two_dim = sample[1]

    num_keypoints = labels_two_dim.sum().item()
    assert num_keypoints >= 0

    if num_keypoints > 0:
        nonzero_indices = torch.nonzero(labels_two_dim)
        assert len(nonzero_indices) == num_keypoints


def prepare_tensor(tensor: Tensor) -> Tensor:
    """Подготавливает тензор для визуализации"""
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor


def tensor_to_image(tensor: Tensor) -> Array:
    """Конвертирует тензор в numpy array для imshow"""
    return tensor.squeeze(0).numpy()


def normalize_label(img_np: Array) -> Array:
    """Нормализует изображение метки"""
    if img_np.max() > img_np.min():
        return (img_np - img_np.min()) / (img_np.max() - img_np.min() + EPS)
    return img_np


def save_image_tensor(tensor: Tensor, filename: Path | str, is_label: bool = False) -> bool:
    """Основная функция сохранения"""
    tensor = prepare_tensor(tensor)
    img_np = tensor_to_image(tensor)

    plt.figure(figsize=(6, 6))

    if is_label:
        plt.imshow(img_np, cmap=GRAY_CMAP, vmin=0, vmax=1)
    else:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + EPS)
        plt.imshow(img_np, cmap=GRAY_CMAP, vmin=0, vmax=1)

    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()
    return True


def to_image(tensor: Tensor) -> Array:
    return (255 * tensor).numpy().astype(np.uint8)


def test_image_saving(synthetic_dataset: SyntheticDataset) -> None:
    """Тестирует сохранение изображений"""
    output_dir = make_dir("tmp/test_output")

    dataloader = torch.utils.data.DataLoader(synthetic_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    img_batch = batch[0]
    mask_batch = batch[1]
    labels_batch = batch[2]

    img = img_batch[0].squeeze(0)
    mask = mask_batch[0].squeeze(0)
    labels = labels_batch[0].squeeze(0)

    plt.imsave(output_dir / "image.png", to_image(img), cmap=GRAY_CMAP)
    plt.imsave(output_dir / "mask.png", to_image(mask), cmap=GRAY_CMAP)
    plt.imsave(output_dir / "labels.png", to_image(labels), cmap=GRAY_CMAP)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
