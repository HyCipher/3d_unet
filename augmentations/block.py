import numpy as np


def random_block_3d(img, label, prob=0.2, shift=50):
    """
    随机四象限亮度偏移增强（仅作用于图像，不改变标签）。
    """
    if np.random.rand() >= prob:
        return img, label

    cy = img.shape[1] // 2
    cx = img.shape[2] // 2
    yloc = np.random.randint(max(cy - shift, 1), min(cy + shift, img.shape[1] - 1))
    xloc = np.random.randint(max(cx - shift, 1), min(cx + shift, img.shape[2] - 1))

    img = img.copy().astype("float32")
    img[:, :yloc, :xloc] -= (np.random.rand() - 0.5) * 0.5
    img[:, :yloc, xloc:] -= (np.random.rand() - 0.5) * 0.5
    img[:, yloc:, xloc:] -= (np.random.rand() - 0.5) * 0.5
    img[:, yloc:, :xloc] -= (np.random.rand() - 0.5) * 0.5
    img = np.clip(img, 0, 1)

    return img, label