import numpy as np


def random_block_3d(img, label, prob=0.2, shift=50):
    """随机四象限亮度偏移增强（仅作用于图像，不改变标签）。

    以图像 Y/X 中心附近的随机点将图像分为四个象限，
    对每个象限施加独立的随机亮度偏移，结果 clip 到 [0, 1]。

    Args:
        img   (np.ndarray): 图像数组，shape 为 (Z, Y, X)。
        label (np.ndarray): 标签数组，shape 为 (Z, Y, X)，不做修改。
        prob  (float): 触发增强的概率。
        shift (int): 分割点在轴中心附近的随机偏移范围（像素）。

    Returns:
        img, label (np.ndarray, np.ndarray)
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