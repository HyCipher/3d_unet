import numpy as np


def random_translate_3d(img, label, prob=0.5):
    """随机平移裁剪增强，同时作用于图像和标签。

    在 Y/X 平面上对中心区域施加随机偏移并裁剪，输出尺寸为原始 Y/X 的一半。
    Z 轴不做裁剪。

    Args:
        img   (np.ndarray): 图像数组，shape 为 (Z, Y, X)，要求 Y == X。
        label (np.ndarray): 标签数组，shape 为 (Z, Y, X)。
        prob  (float): 触发增强的概率。

    Returns:
        img, label (np.ndarray, np.ndarray)，Y/X 尺寸缩减为原来的一半。
    """
    if np.random.rand() >= prob:
        return img, label

    h = img.shape[1]   # Y
    w = img.shape[2]   # X
    out_h = h // 2
    out_w = w // 2

    tw = np.random.randint(-out_h // 2, out_h // 2)
    th = np.random.randint(-out_w // 2, out_w // 2)

    cy = h // 2 - tw
    cx = w // 2 - th

    img   = img[:,   cy:cy + out_h, cx:cx + out_w]
    label = label[:, cy:cy + out_h, cx:cx + out_w]

    return img, label