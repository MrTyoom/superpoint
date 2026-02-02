import shutil
import random
from pathlib import Path

import numpy as np
import torch


def make_dir(path):
    path = Path(path)

    if path.is_dir():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
