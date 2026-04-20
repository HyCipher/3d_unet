import numpy as np


def random_flip_3d(img, label, prob=0.5):
    """随机翻转。"""
    # Flip along the Z-axis
    if np.random.rand() < prob:
        img = np.flip(img, axis=0).copy()
        label = np.flip(label, axis=0).copy()
        
    # Flip along the Y-axis
    # if np.random.rand() < prob:
    #     img = np.flip(img, axis=1).copy()
    #     label = np.flip(label, axis=1).copy()

    # Flip along the X-axis
    if np.random.rand() < prob:
        img = np.flip(img, axis=2).copy()
        label = np.flip(label, axis=2).copy()

    return img, label