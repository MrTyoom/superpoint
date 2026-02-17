import matplotlib
import pytest
import torch

matplotlib.use("Agg")
from pathlib import Path

# noqa: WPS301
import matplotlib.pyplot as plt
import rootutils
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from torch.types import Tensor

root = rootutils.setup_root(__file__, indicator="src", pythonpath=True)
EPS = 1e-8

from src.synthetic_loader import SyntheticDataset


@pytest.fixture(scope="module")
def config() -> DictConfig:
    """Загружает конфигурацию для тестов"""
    cfg = OmegaConf.load(root / "params.yaml")
    return cfg["train_magicpoint"]


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

    assert all(isinstance(sample_el, Tensor) for sample_el in sample)

    labels_two_dim_squeezed = labels_two_dim.squeeze()
    valid_mask_squeezed = valid_mask.squeeze()

    assert labels_two_dim_squeezed.dim() == 2
    assert valid_mask_squeezed.dim() == 2

    height = labels_two_dim_squeezed.shape[0]
    width = labels_two_dim_squeezed.shape[1]

    assert valid_mask_squeezed.shape == (height, width)

    cfg = OmegaConf.load(root / "params.yaml")
    expected_size = cfg["prepare_synthetic_data"]["image_size"]

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


def tensor_to_image(tensor: Tensor) -> Tensor:
    """Конвертирует тензор в numpy array для imshow"""
    if tensor.dim() == 2:
        return tensor.numpy()
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:  # (1, H, W)
            return tensor.squeeze(0).numpy()
        if tensor.shape[2] == 1:  # (H, W, 1)
            return tensor.squeeze(2).numpy()
        if tensor.shape[0] == 3:  # (3, H, W) RGB
            return tensor.permute(1, 2, 0).numpy()
    return tensor.mean(dim=0).numpy()  # усреднение по каналам


def normalize_label(img_np: NDArray) -> NDArray:
    """Нормализует изображение метки"""
    if img_np.max() > img_np.min():
        return (img_np - img_np.min()) / (img_np.max() - img_np.min() + EPS)
    return img_np


def save_image_tensor(tensor: Tensor, filename: Path | str, is_label: bool = False) -> bool:
    """Основная функция сохранения"""
    tensor = prepare_tensor(tensor)
    img_np = tensor_to_image(tensor)
    if is_label:
        img_np = normalize_label(img_np)

    plt.figure(figsize=(6, 6))
    if img_np.ndim == 2:
        plt.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(img_np)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()
    return True


def test_image_saving(tmp_path: Path, synthetic_dataset: SyntheticDataset) -> None:
    """Тестирует сохранение изображений"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    dataloader = torch.utils.data.DataLoader(synthetic_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    img_batch = batch[0]
    mask_batch = batch[1]
    labels_batch = batch[2]

    idx = 0
    assert save_image_tensor(img_batch[idx], output_dir / "original_img.png")
    assert save_image_tensor(labels_batch[idx], output_dir / "labels_2D.png", is_label=True)
    assert save_image_tensor(mask_batch[idx], output_dir / "valid_mask.png", is_label=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
