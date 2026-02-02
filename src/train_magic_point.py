from pathlib import Path

import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True) # noqa

from src.utils import seed_everything


def main(cfg):
    seed_everything(cfg.seed)


if __name__ == "__main__":
    
    params = Path(__file__).stem
    cfg = OmegaConf.load("params.yaml")[params]

    main(cfg)