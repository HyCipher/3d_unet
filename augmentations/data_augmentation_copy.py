from .rotate import random_rotation_90_3d
from .flip import random_flip_3d
from .gaussian_noise import random_gaussian_noise
from .contrast import random_contrast_3d
from .blackpad import random_blackpad_3d
from .block import random_block_3d
from .darkline import random_darkline_3d
from .elastic import random_elastic_deformation_3d
from .translate import random_translate_3d
from .missing_section import random_missing_section
from .section_intensity_shift import random_section_intensity_shift


def apply_augmentation(img, label, augment=True):
    """统一的增强接口。"""
    if not augment:
        return img, label

    img, label = random_flip_3d(img, label, prob=0.5)
    img, label = random_rotation_90_3d(img, label, prob=0.5)           # EM XY 平面各向同性，90°旋转完全合理
    img, label = random_contrast_3d(img, label, prob=0.3,
                                    contrast_range=(0.5, 1.5), brightness_range=(-0.25, 0.25), gamma_log2_range=(-1.0, 1.0)
                                    )
    img, label = random_gaussian_noise(img, label, prob=0.2, std=0.02) # EM 图像噪声显著，std=0.01 无实际效果
    # img, label = random_blackpad_3d(img, label, prob=0.25, pad_ratio_range=(0.4, 0.9))
    # img, label = random_block_3d(img, label, prob=0.2, shift=50)
    # img, label = random_elastic_deformation_3d(img, label, prob=0.3, alpha=40.0, sigma=6.0)  # alpha 提高：组织包埋/切片形变更大
    # img, label = random_darkline_3d(img, label, prob=0.4, width_range=(5, 25))               # 刀痕(knife mark)是 EM 高频伪影
    # img, label = random_missing_section(img, label, prob=0.3, max_missing=2)                 # 缺失切片：串行切片 EM 常见伪影
    img, label = random_section_intensity_shift(img, label, prob=0.4, std=0.1)               # 逐切片染色/焦距漂移
    img, label = random_translate_3d(img, label, prob=0.3, min_shift=(2, 2, 2), max_shift=(5, 5, 5))

    return img, label