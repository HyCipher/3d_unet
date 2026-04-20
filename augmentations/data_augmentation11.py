"""Main data augmentation pipeline."""

import numpy as np
from .flip import random_flip_3d
from .rotate import random_rotation_90_3d
from .gaussian_noise import random_gaussian_noise
from .contrast import random_contrast_3d


def apply_augmentation(x_patch, y_patch, augment=True, prob=0.5):
    """
    Apply random data augmentation to 3D patches.
    
    Args:
        x_patch: 3D input patch (D, H, W)
        y_patch: 3D label patch (D, H, W)
        augment: whether to apply augmentation
        prob: probability threshold for each augmentation
    
    Returns:
        x_patch_aug, y_patch_aug: augmented patches
    """
    if not augment:
        return x_patch, y_patch
    
    # Apply augmentations with probability
    if np.random.rand() < prob:
        x_patch, y_patch = random_flip_3d(x_patch, y_patch)
    
    if np.random.rand() < prob:
        x_patch, y_patch = random_rotation_90_3d(x_patch, y_patch)
    
    if np.random.rand() < prob:
        x_patch = random_gaussian_noise(x_patch)
    
    if np.random.rand() < prob:
        x_patch = random_contrast_3d(x_patch)
    
    return x_patch, y_patch
