"""photometric augmentation
# used in dataloader
"""

import cv2
import numpy as np
from imgaug import augmenters as iaa
from numpy.random import randint


class ImgAugTransform:
    def __init__(self, 
                 random_brightness=75,
                 random_contrast=[0.3, 1.8],
                 additive_gaussian_noise=[0, 15],
                 additive_speckle_noise=[0, 0.0035],
                 motion_blur=7,
                 **config
        ):
        ## old photometric
        self.aug = iaa.Sequential(
            [
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                iaa.Sometimes(
                    0.25,
                    iaa.OneOf(
                        [
                            iaa.Dropout(p=(0, 0.1)),
                            iaa.CoarseDropout(0.1, size_percent=0.5),
                        ]
                    ),
                ),
                iaa.Sometimes(
                    0.25,
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
                ),
            ]
        )


        if config["photometric"]["enable"]:
            aug_all = []
            if random_brightness:
                aug = iaa.Add((random_brightness, random_brightness))
                aug_all.append(aug)

            if random_contrast:
                aug = iaa.LinearContrast((random_contrast[0], random_contrast[1]))
                aug_all.append(aug)

            if additive_gaussian_noise:
                aug = iaa.AdditiveGaussianNoise(scale=(additive_gaussian_noise[0],
                                                       additive_gaussian_noise[1]))
                aug_all.append(aug)

            if additive_speckle_noise:
                aug = iaa.ImpulseNoise(p=(additive_speckle_noise[0],
                                          additive_speckle_noise[1]))
                aug_all.append(aug)

            if motion_blur:
                if motion_blur > 3:
                    motion_blur = randint(3, motion_blur)
                elif motion_blur == 3:
                    aug = iaa.Sometimes(0.5, iaa.MotionBlur(motion_blur))
                aug_all.append(aug)


            self.aug = iaa.Sequential(aug_all)

        else:
            self.aug = iaa.Sequential(
                [
                    iaa.Noop(),
                ]
            )

    def __call__(self, img):
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        img = self.aug.augment_image(img)
        img = img.astype(np.float32) / 255
        return img


class customizedTransform:
    def __init__(self):
        pass

    def additive_shade(
        self,
        image,
        nb_ellipses=20,
        transparency_range=[-0.5, 0.8],
        kernel_size_range=[50, 100],
    ):
        def _py_additive_shade(img):
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)
            for _ in range(nb_ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))
                max_rad = max(ax, ay)
                x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
                y = np.random.randint(max_rad, img.shape[0] - max_rad)
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

            transparency = np.random.uniform(*transparency_range)
            kernel_size = np.random.randint(*kernel_size_range)
            if (kernel_size % 2) == 0:  # kernel_size has to be odd
                kernel_size += 1
            mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)

            shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.0)
            return np.clip(shaded, 0, 255)

        shaded = _py_additive_shade(image)
        return shaded

    def __call__(self, img):
        img = self.additive_shade(img * 255)
        return img / 255
