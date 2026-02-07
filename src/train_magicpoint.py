import rootutils
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.logger import LOG
from src.synthetic_loader import Loader


def main(cfg: DictConfig) -> None:
    LOG.info("set seed: {0}".format(cfg.seed))
    seed_everything(cfg.seed)

    loader = Loader(cfg)
    loader.setup(stage="fit")

    train_loader = loader.train_dataloader()
    LOG.info("train loader size: {0}".format(len(train_loader)))

    val_loader = loader.val_dataloader()
    LOG.info("val loader size: {0}".format(len(val_loader)))


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["train_magicpoint"]

    main(cfg)
