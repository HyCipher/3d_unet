import random

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

    # Keep augmentation simple: at most one geometric op + one intensity op.
    geometric_ops = (
        lambda image, target: random_flip_3d(image, target, prob=1.0),
        lambda image, target: random_rotation_90_3d(image, target, prob=1.0),
        lambda image, target: random_translate_3d(
            image,
            target,
            prob=1.0,
            min_shift=(0, 1, 1),
            max_shift=(0, 3, 3),
        ),
    )
    intensity_ops = (
        lambda image, target: random_contrast_3d(
            image,
            target,
            prob=1.0,
            contrast_range=(0.8, 1.2),
            brightness_range=(-0.08, 0.08),
            gamma_log2_range=(-0.3, 0.3),
        ),
        lambda image, target: random_gaussian_noise(image, target, prob=1.0, std=0.03),
        lambda image, target: random_section_intensity_shift(image, target, prob=1.0, std=0.05),
    )
    artifact_ops = (
        lambda image, target: random_darkline_3d(image, target, prob=1.0, width_range=(5, 12)),
        lambda image, target: random_missing_section(image, target, prob=1.0, max_missing=1),
        lambda image, target: random_elastic_deformation_3d(
            image,
            target,
            prob=1.0,
            alpha=8.0,
            sigma=8.0,
        ),
    )

    if random.random() < 0.8:
        img, label = random.choice(geometric_ops)(img, label)

    if random.random() < 0.8:
        img, label = random.choice(intensity_ops)(img, label)

    # Rare single artifact branch for EM-specific acquisition issues.
    if random.random() < 0.1:
        img, label = random.choice(artifact_ops)(img, label)

    return img, label