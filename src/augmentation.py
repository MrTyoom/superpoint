from imgaug import augmenters as iaa
import numpy as np
import cv2
from numpy.random import randint


class ImgAugTransform:
    def __init__(self, config):
        self.config = config

        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.25, iaa.OneOf([iaa.Dropout(p=(0, 0.1)), iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5))
        ])

        if config['enable']:
            params = config['params']
            aug_all = []

            if params.get('random_brightness', False):
                change = params['random_brightness']['max_abs_change']
                aug = iaa.Add((-change, change))
                aug_all.append(aug)

            if params.get('random_contrast', False):
                change = params['random_contrast']['strength_range']
                aug = iaa.LinearContrast((change[0], change[1]))
                aug_all.append(aug)
        
            if params.get('additive_gaussian_noise', False):
                change = params['additive_gaussian_noise']['stddev_range']
                aug = iaa.AdditiveGaussianNoise(scale=(change[0], change[1]))
                aug_all.append(aug)
           
            if params.get('additive_speckle_noise', False):
                change = params['additive_speckle_noise']['prob_range']
                aug = iaa.ImpulseNoise(p=(change[0], change[1]))
                aug_all.append(aug)
           
            if params.get('motion_blur', False):
                change = params['motion_blur']['max_kernel_size']
                
                if change > 3:
                    change = randint(3, change)
                elif change == 3:
                    aug = iaa.Sometimes(0.5, iaa.MotionBlur(change))

                aug_all.append(aug)

            if params.get('GaussianBlur', False):
                change = params['GaussianBlur']['sigma']
                aug = iaa.GaussianBlur(sigma=(change))
                aug_all.append(aug)

            self.aug = iaa.Sequential(aug_all)
        else:
            self.aug = iaa.Sequential([iaa.Noop()])

    def additive_shade(self, image, nb_ellipses=20, transparency_range=[-0.5, 0.8], kernel_size_range=[250, 350]):
        def _py_additive_shade(img):
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)

            for i in range(nb_ellipses):
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
            shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.)

            shaded = np.clip(shaded, 0, 255)
            image = shaded[:, :].astype(np.uint8)

            return image

        image = _py_additive_shade(image[..., None])

        return image

    def __call__(self, img):
        img = self.aug.augment_image(img)
        
        if self.config['params']['additive_shade']:
            params = self.config['params']['additive_shade']
            img = self.additive_shade(img, **params)

        return img