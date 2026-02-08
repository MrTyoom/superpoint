from omegaconf import DictConfig

from src.augment_helper import additive_shade, generate_mask, get_photometric_augmentation


class PhotometricAugmentation:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.aug = get_photometric_augmentation(cfg)

    def __call__(self, img):
        img = self.aug.augment_image(img)

        cfg = self.cfg.additive_shade
        mask = generate_mask(cfg, img.shape[:2])
        img = additive_shade(cfg, mask, img)

        return img
