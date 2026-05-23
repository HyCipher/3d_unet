import numpy as np


def random_darkline_3d(img, label, prob=0.2, width_range=(10, 20)):
    """
    随机暗带增强（仅作用于图像，不改变标签）。
    """
    if np.random.rand() >= prob:
        return img, label

    w = np.random.randint(width_range[0], width_range[1] + 1)
    b = np.abs(np.random.rand() - 0.5) * 0.5

    img = img.copy().astype("float32")

    if np.random.rand() < 0.5:
        # X 轴方向暗带
        l = img.shape[2]
        loc = np.random.randint(w + 5, l - (w + 5))
        img[:, :, loc:loc + w] -= b
    else:
        # Y 轴方向暗带
        l = img.shape[1]
        loc = np.random.randint(w + 5, l - (w + 5))
        img[:, loc:loc + w, :] -= b

    img = np.clip(img, 0, 1)

    return img, label