import numpy as np


def random_blackpad_3d(img, label, prob=0.25, pad_ratio_range=(0.4, 0.9)):
    """随机黑边填充增强，沿 Y 轴或 X 轴方向平移并以黑色填充空白区域。

    Args:
        img   (np.ndarray): 图像数组，shape 为 (Z, Y, X)。
        label (np.ndarray): 标签数组，shape 为 (Z, Y, X)。
        prob  (float): 每个方向上触发增强的概率。
        pad_ratio_range (tuple): 平移量占轴长的比例范围 (min, max)。

    Returns:
        img, label (np.ndarray, np.ndarray)
    """
    lo, hi = pad_ratio_range

    # Y 轴方向
    if np.random.rand() < prob:
        w = img.shape[1]
        n = np.random.randint(int(w * lo), int(w * hi) + 1)
        d = np.random.choice([0, 1])
        img   = _blackpad(img,   d=d, n=n)
        label = _blackpad(label, d=d, n=n)

    # X 轴方向
    if np.random.rand() < prob:
        w = img.shape[2]
        n = np.random.randint(int(w * lo), int(w * hi) + 1)
        d = np.random.choice([2, 3])
        img   = _blackpad(img,   d=d, n=n)
        label = _blackpad(label, d=d, n=n)

    return img, label


def _blackpad(arr, d=0, n=100):
    """将数组沿指定方向平移 n 个像素，空白处填零。

    Args:
        arr (np.ndarray): shape (Z, Y, X)。
        d   (int): 方向 — 0: Y 正方向, 1: Y 负方向,
                           2: X 正方向, 3: X 负方向。
        n   (int): 平移像素数。

    Returns:
        np.ndarray
    """
    out = np.zeros(arr.shape, dtype="float32")

    if d == 0:
        out[:, n:, :] = arr[:, :-n, :]
    elif d == 1:
        out[:, :-n, :] = arr[:, n:, :]
    elif d == 2:
        out[:, :, n:] = arr[:, :, :-n]
    elif d == 3:
        out[:, :, :-n] = arr[:, :, n:]

    return out