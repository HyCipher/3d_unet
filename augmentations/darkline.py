import numpy as np


def random_darkline_3d(img, label, prob=0.2, width_range=(10, 20)):
    """随机暗带增强（仅作用于图像，不改变标签）。

    沿 Y 轴或 X 轴随机选取一条带状区域，对其施加亮度衰减，结果 clip 到 [0, 1]。

    Args:
        img         (np.ndarray): 图像数组，shape 为 (Z, Y, X)。
        label       (np.ndarray): 标签数组，shape 为 (Z, Y, X)，不做修改。
        prob        (float): 触发增强的概率。
        width_range (tuple): 暗带宽度的随机范围（像素）。

    Returns:
        img, label (np.ndarray, np.ndarray)
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