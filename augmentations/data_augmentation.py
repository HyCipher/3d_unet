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
from config.aug_config import get_aug_config

_AUG_CFG = get_aug_config()


def apply_augmentation(img, label, augment=True):
    """Apply a random combination of augmentations to the image and label."""
    if not augment:
        return img, label

    cfg = _AUG_CFG

    # Define augmentation groups, each containing multiple specific operations.
    # During augmentation, one operation is randomly selected from each group.
    geometric_ops = (
        lambda image, target: random_flip_3d(image, target, prob=1.0),
        lambda image, target: random_rotation_90_3d(image, target, prob=1.0),
        lambda image, target: random_translate_3d(
            image,
            target,
            prob=1.0,
            min_shift=cfg["translate_min_shift"],
            max_shift=cfg["translate_max_shift"],
        ),
        lambda image, target: random_blackpad_3d(
            image,
            target,
            prob=cfg["blackpad_prob"],
            pad_ratio_range=cfg["blackpad_pad_ratio_range"],
        ),
    )
    
    intensity_ops = (
        lambda image, target: random_contrast_3d(
            image,
            target,
            prob=1.0,
            contrast_range=cfg["contrast_range"],
            brightness_range=cfg["brightness_range"],
            gamma_log2_range=cfg["gamma_log2_range"],
        ),
        lambda image, target: random_gaussian_noise(image, target, prob=1.0, std=cfg["gaussian_noise_std"]),
        lambda image, target: random_section_intensity_shift(image, target, prob=1.0, std=cfg["section_intensity_shift_std"]),
        lambda image, target: random_block_3d(image, target, prob=cfg["block_prob"], shift=cfg["block_shift"]),
    )
    
    artifact_ops = (
        lambda image, target: random_darkline_3d(image, target, prob=1.0, width_range=cfg["darkline_width_range"]),
        lambda image, target: random_missing_section(image, target, prob=1.0, max_missing=cfg["missing_section_max"]),
        lambda image, target: random_elastic_deformation_3d(
            image,
            target,
            prob=1.0,
            alpha=cfg["elastic_alpha"],
            sigma=cfg["elastic_sigma"],
        ),
    )

    if random.random() < cfg["prob_geometric"]:
        img, label = random.choice(geometric_ops)(img, label)

    if random.random() < cfg["prob_intensity"]:
        img, label = random.choice(intensity_ops)(img, label)

    # Rare single artifact branch for EM-specific acquisition issues.
    if random.random() < cfg["prob_artifact"]:
        img, label = random.choice(artifact_ops)(img, label)

    return img, label