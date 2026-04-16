import numpy as np


def random_rotation_90_3d(img, label, prob=0.5):
    """随机90度旋转（XY平面）。"""
    print("🔄 Applying random_rotation_90_3d augmentation...")
    if np.random.rand() < prob:
        k = np.random.randint(1, 4)
        img = np.rot90(img, k, axes=(1, 2)).copy()
        label = np.rot90(label, k, axes=(1, 2)).copy()

    return img, label