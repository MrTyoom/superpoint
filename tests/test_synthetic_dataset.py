import matplotlib
import pytest
import torch

matplotlib.use("Agg")
from pathlib import Path

# noqa: WPS301
import matplotlib.pyplot as plt
import rootutils
from omegaconf import OmegaConf

root = rootutils.setup_root(__file__, indicator="src", pythonpath=True)
EPS = 1e-8

from src.synthetic_loader import DataSet


@pytest.fixture(scope="module")
def config():
    """Загружает конфигурацию для тестов"""
    cfg = OmegaConf.load(root / "params.yaml")
    return cfg["train_magicpoint"]


@pytest.fixture(scope="module")
def synthetic_dataset(config):
    """Создает датасет для тестов"""
    cfg = config.copy()

    data_path = root / cfg["data_dir"]
    files = list(Path(data_path).glob("**/*.png"))
    files = [f for f in files if f.with_suffix(".npy").exists()]
    return DataSet(files, cfg["augmentation"], device="cpu")


def test_dataset_item_shape(synthetic_dataset):
    """Тестирует форму элементов датасета (6 элементов)"""
    sample = synthetic_dataset[0]
    assert len(sample) == 6, f"Expected 6 items, got {len(sample)}"

    labels_two_dim = sample[1]
    valid_mask = sample[2]
    warped_img = sample[3]
    warped_labels_res = sample[4]
    homography = sample[5]

    assert all(isinstance(t, torch.Tensor) for t in sample)

    labels_two_dim_squeezed = labels_two_dim.squeeze()
    valid_mask_squeezed = valid_mask.squeeze()

    assert labels_two_dim_squeezed.dim() == 2
    assert valid_mask_squeezed.dim() == 2
    assert warped_labels_res.dim() == 3

    assert homography.shape == (3, 3), f"homography should be (3,3), got {homography.shape}"

    H = labels_two_dim_squeezed.shape[0]
    W = labels_two_dim_squeezed.shape[1]

    assert valid_mask_squeezed.shape == (H, W)
    assert warped_labels_res.shape[1:] == (H, W)
    assert warped_img.shape[1:] == (H, W)

    cfg = OmegaConf.load(root / "params.yaml")
    expected_size = cfg["prepare_synthetic_data"]["image_size"]

    assert (H, W) == tuple(expected_size) or (W, H) == tuple(expected_size)


def test_dataloader_works(synthetic_dataset):
    """Тестирует работу DataLoader с датасетом"""
    dataloader = torch.utils.data.DataLoader(
        synthetic_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True
    )

    batch = next(iter(dataloader))
    assert len(batch) == 6, f"Expected 6 items in batch, got {len(batch)}"

    img_batch = batch[0]
    labels_two_dim_batch = batch[1]
    valid_mask_batch = batch[2]
    warped_img_batch = batch[3]
    warped_labels_res_batch = batch[4]
    homography_batch = batch[5]

    assert img_batch.shape[0] == 4
    assert labels_two_dim_batch.shape[0] == 4
    assert valid_mask_batch.shape[0] == 4
    assert warped_img_batch.shape[0] == 4
    assert warped_labels_res_batch.shape[0] == 4
    assert homography_batch.shape[0] == 4
    assert homography_batch.shape[1:] == (3, 3)


def test_keypoints_statistics(synthetic_dataset):
    """Тестирует статистику ключевых точек"""
    sample = synthetic_dataset[0]
    labels_two_dim = sample[1]

    num_keypoints = labels_two_dim.sum().item()
    assert num_keypoints >= 0

    if num_keypoints > 0:
        nonzero_indices = torch.nonzero(labels_two_dim)
        assert len(nonzero_indices) == num_keypoints


def prepare_tensor(tensor):
    """Подготавливает тензор для визуализации"""
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor


def tensor_to_image(tensor):
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


def normalize_label(img_np):
    """Нормализует изображение метки"""
    if img_np.max() > img_np.min():
        return (img_np - img_np.min()) / (img_np.max() - img_np.min() + EPS)
    return img_np


def save_image_tensor(tensor, filename, is_label=False):
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


def test_image_saving(tmp_path, synthetic_dataset):
    """Тестирует сохранение изображений"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    dataloader = torch.utils.data.DataLoader(synthetic_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    img_batch = batch[0]
    labels_batch = batch[1]
    mask_batch = batch[2]
    warped_batch = batch[3]
    res_batch = batch[4]

    idx = 0
    assert save_image_tensor(img_batch[idx], output_dir / "original_img.png")
    assert save_image_tensor(labels_batch[idx], output_dir / "labels_2D.png", is_label=True)
    assert save_image_tensor(mask_batch[idx], output_dir / "valid_mask.png", is_label=True)
    assert save_image_tensor(warped_batch[idx], output_dir / "warped_img.png")

    warped_res = res_batch[idx]
    assert save_image_tensor(warped_res[0], output_dir / "warped_labels_res_x.png", is_label=True)
    assert save_image_tensor(warped_res[1], output_dir / "warped_labels_res_y.png", is_label=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
