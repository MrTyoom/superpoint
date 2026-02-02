import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

import rootutils

root = rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.datasets.SyntheticDataset import SyntheticDataset


@pytest.fixture(scope="module")
def config():
    """Загружает конфигурацию для тестов"""
    cfg = OmegaConf.load(root / "src" / "params.yaml")
    return cfg['prepare_synthetic_dataset']


@pytest.fixture(scope="module")
def synthetic_dataset(config):
    """Создает датасет для тестов"""
    cfg = config.copy()
    return SyntheticDataset(**cfg)


def test_dataset_creation(synthetic_dataset):
    """Тестирует создание датасета"""
    assert len(synthetic_dataset) > 0
    assert hasattr(synthetic_dataset, 'data')
    assert isinstance(synthetic_dataset.data, list)
    assert len(synthetic_dataset.data) > 0


def test_dataset_item_shape(synthetic_dataset):
    """Тестирует форму элементов датасета"""
    item = synthetic_dataset[0]
    assert len(item) == 5
    
    img, labels_2D, valid_mask, warped_img, warped_labels_res = item
    
    assert all(isinstance(t, torch.Tensor) for t in item)
    
    labels_2D_squeezed = labels_2D.squeeze()
    valid_mask_squeezed = valid_mask.squeeze()
    
    assert labels_2D_squeezed.dim() == 2
    assert valid_mask_squeezed.dim() == 2
 
    assert warped_labels_res.dim() == 3
    assert warped_labels_res.shape[0] == 2
  
    H = labels_2D_squeezed.shape[0]
    W = labels_2D_squeezed.shape[1]

    assert valid_mask_squeezed.shape == (H, W)
    assert warped_labels_res.shape[1:] == (H, W)
    
    
    cfg = OmegaConf.load(root / "src" / "params.yaml")
    expected_size = cfg['prepare_synthetic_dataset']['image_size']
    
    assert (H, W) == tuple(expected_size) or (W, H) == tuple(expected_size)


def test_dataloader_works(synthetic_dataset):
    """Тестирует работу DataLoader с датасетом"""
    dataloader = torch.utils.data.DataLoader(
        synthetic_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    batch = next(iter(dataloader))
    img_batch, labels_2D_batch, valid_mask_batch, warped_img_batch, warped_labels_res_batch = batch
    
    assert img_batch.shape[0] == 4
    assert labels_2D_batch.shape[0] == 4
    assert valid_mask_batch.shape[0] == 4
    assert warped_img_batch.shape[0] == 4
    assert warped_labels_res_batch.shape[0] == 4


def test_warped_labels_res_range(synthetic_dataset):
    """Тестирует диапазон субпиксельных смещений"""
    item = synthetic_dataset[0]
    _, _, _, _, warped_labels_res = item
    
    assert -0.5 <= warped_labels_res.min() <= warped_labels_res.max() <= 0.5


def test_keypoints_statistics(synthetic_dataset):
    """Тестирует статистику ключевых точек"""
    item = synthetic_dataset[0]
    _, labels_2D, _, _, _ = item
    
    num_keypoints = labels_2D.sum().item()
    assert num_keypoints >= 0 
    
    if num_keypoints > 0:
        nonzero_indices = torch.nonzero(labels_2D)
        assert len(nonzero_indices) == num_keypoints


def test_image_saving(tmp_path, synthetic_dataset):
    """Тестирует сохранение изображений"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    
    def save_image_tensor(tensor, filename, is_label=False):
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        if tensor.dim() == 3:
            if tensor.shape[0] == 1:
                img_np = tensor.squeeze(0).numpy()
            elif tensor.shape[0] == 3:
                img_np = tensor.permute(1, 2, 0).numpy()
            elif tensor.shape[2] == 1:
                img_np = tensor.squeeze(2).numpy()
            else:
                img_np = tensor.mean(dim=0).numpy()
        elif tensor.dim() == 2:
            img_np = tensor.numpy()
        else:
            return False
        
        if is_label and img_np.max() > img_np.min():
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        plt.figure(figsize=(6, 6))
        if img_np.ndim == 2:
            plt.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(img_np)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True
    
    dataloader = torch.utils.data.DataLoader(
        synthetic_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    batch = next(iter(dataloader))
    img_batch, labels_2D_batch, valid_mask_batch, warped_img_batch, warped_labels_res_batch = batch
    
    idx = 0
    
    assert save_image_tensor(img_batch[idx], output_dir / 'original_img.png')
    assert save_image_tensor(labels_2D_batch[idx], output_dir / 'labels_2D.png', is_label=True)
    assert save_image_tensor(valid_mask_batch[idx], output_dir / 'valid_mask.png', is_label=True)
    assert save_image_tensor(warped_img_batch[idx], output_dir / 'warped_img.png')
    
    warped_res = warped_labels_res_batch[idx]
    assert save_image_tensor(warped_res[0], output_dir / 'warped_labels_res_x.png', is_label=True)
    assert save_image_tensor(warped_res[1], output_dir / 'warped_labels_res_y.png', is_label=True)

def test_multiple_samples(synthetic_dataset):
    """Тестирует несколько сэмплов датасета"""
    for idx in [0, 10, 100]:
        if idx < len(synthetic_dataset):
            item = synthetic_dataset[idx]
            assert len(item) == 5
            
            for tensor in item:
                assert isinstance(tensor, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])