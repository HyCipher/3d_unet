import numpy as np


def random_gaussian_noise(img, label, prob=0.1, std=0.01):
    """添加高斯噪声。"""
    if np.random.rand() < prob:
        noise = np.random.normal(0, std, img.shape)
        img = img + noise

    return img, label