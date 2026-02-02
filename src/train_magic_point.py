from pathlib import Path

import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True) # noqa

from src.utils import seed_everything
from src.dataset import get_loaders


def main(cfg):
    seed_everything(cfg.seed)

    train_loader, test_loader = get_loaders(cfg)

    print(f"len(train_loader) = {len(train_loader)}")
    print(f"len(test_loader) = {len(test_loader)}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


if __name__ == "__main__":
    
    params = Path(__file__).stem
    cfg = OmegaConf.load("params.yaml")[params]

    main(cfg)