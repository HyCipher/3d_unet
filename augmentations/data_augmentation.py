import numpy as np
from .rotate import random_rotation_90_3d
from .flip import random_flip_3d
from .gaussian_noise import random_gaussian_noise
from .contrast import random_contrast_3d


def apply_augmentation(img, label, augment=True):
    """统一的增强接口。"""
    if not augment:
        return img, label

    img, label = random_flip_3d(img, label, prob=0.8)
    img, label = random_rotation_90_3d(img, label, prob=0.8)
    img, label = random_contrast_3d(img, label, prob=0.3,
                                    contrast_range=(0.5, 1.5),brightness_range=(-0.25, 0.25), gamma_log2_range=(-1.0, 1.0)
                                    )
    img, label = random_gaussian_noise(img, label, prob=0.1, std=0.01)

    return img, label