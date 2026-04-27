from .rotate import random_rotation_90_3d
from .flip import random_flip_3d
from .gaussian_noise import random_gaussian_noise
from .contrast import random_contrast_3d
from .blackpad import random_blackpad_3d
from .block import random_block_3d
from .darkline import random_darkline_3d
from .translate import random_translate_3d


def apply_augmentation(img, label, augment=True):
    """统一的增强接口。"""
    if not augment:
        return img, label

    img, label = random_flip_3d(img, label, prob=0.8)
    img, label = random_rotation_90_3d(img, label, prob=0.8)
    img, label = random_contrast_3d(img, label, prob=0.5,
                                    contrast_range=(0.5, 1.5),brightness_range=(-0.25, 0.25), gamma_log2_range=(-1.0, 1.0)
                                    )
    img, label = random_gaussian_noise(img, label, prob=0.15, std=0.01)
    img, label = random_blackpad_3d(img, label, prob=0.25, pad_ratio_range=(0.4, 0.9))
    img, label = random_block_3d(img, label, prob=0.2, shift=50)
    img, label = random_darkline_3d(img, label, prob=0.2, width_range=(10, 20))
    img, label = random_translate_3d(img, label, prob=0.5)

    return img, label