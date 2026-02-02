import importlib
import logging
import os
from pathlib import Path

import rootutils
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from omegaconf import OmegaConf

from datasets.SyntheticDataset import SyntheticDataset as Dataset
   

root = rootutils.setup_root(__file__, indicator="src", pythonpath=True)


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """

    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)


def get_module(module_path, attribute_name):
    """
    Динамически импортирует класс или функцию.
    """
    import importlib
    if module_path == '':
        mod = importlib.import_module(module_path)
    else:
        mod = importlib.import_module('{}.{}'.format(module_path, attribute_name))
    return getattr(mod, attribute_name)


def get_save_path(output_dir):
    """
    Функция для сохранения чекпоинтов
    """
    save_path = Path(output_dir) / "checkpoints"
    logging.info(f"=> will save everything to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    return save_path


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def dataLoader(config, dataset="SyntheticDataset", warp_input=False, train=True, val=True):
    # if torch.cuda.is_available():
    #     generator = torch.Generator(device="cuda")
    # else:
    #     generator = torch.Generator(device="cpu")
    generator = torch.Generator(device="cpu")

    
    training_params = config.get("training", {})
    workers_train = training_params.get("workers_train", 0)  # 16
    workers_val = training_params.get("workers_val", 0)  # 16

    logging.info(f"workers_train: {workers_train}, workers_val: {workers_val}")
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    }

    result = {}

    if train:
        train_set = Dataset(
            transform=data_transforms["train"],
            task="train",
            **config,
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config["model"]["batch_size"],
            shuffle=True,
            pin_memory=False,
            num_workers=workers_train,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

        result.update({"train_loader": train_loader, "train_set": train_set})

    if val:
        val_set = Dataset(
            transform=data_transforms["val"],
            task="val",
            **config,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config["model"]["eval_batch_size"],
            shuffle=True,
            pin_memory=False,
            num_workers=workers_val,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        result.update({"val_loader": val_loader, "val_set": val_set})

    if train:
        logging.info(f"Train samples: {len(train_set)}")
    if val:
        logging.info(f"Val samples: {len(val_set)}")

    return result

# if __name__ == '__main__':
#     cfg = OmegaConf.load(root / "src" / "params.yaml")['prepare_synthetic_dataset']
#     loader = dataLoader(cfg)
#     print(loader)