import numpy as np


def _shift_with_padding(arr, dz, dy, dx):
    """Shift array on Z/Y/X axes and pad exposed area with zeros."""
    z, h, w = arr.shape
    out = np.zeros_like(arr)

    if abs(dz) >= z or abs(dy) >= h or abs(dx) >= w:
        return out

    if dz >= 0:
        src_z0, src_z1 = 0, z - dz
        dst_z0, dst_z1 = dz, z
    else:
        src_z0, src_z1 = -dz, z
        dst_z0, dst_z1 = 0, z + dz

    if dy >= 0:
        src_y0, src_y1 = 0, h - dy
        dst_y0, dst_y1 = dy, h
    else:
        src_y0, src_y1 = -dy, h
        dst_y0, dst_y1 = 0, h + dy

    if dx >= 0:
        src_x0, src_x1 = 0, w - dx
        dst_x0, dst_x1 = dx, w
    else:
        src_x0, src_x1 = -dx, w
        dst_x0, dst_x1 = 0, w + dx

    out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return out


def random_translate_3d(img, label, prob=0.2, min_shift=(2, 2, 2), max_shift=(5, 5, 5)):
    """随机平移增强，同时作用于图像和标签，保持尺寸不变。

    Args:
        img (np.ndarray): 图像数组，shape 为 (Z, Y, X)。
        label (np.ndarray): 标签数组，shape 为 (Z, Y, X)。
        prob (float): 触发增强的概率。
        min_shift (tuple): Z/Y/X 三个轴的最小平移体素数（含）。
        max_shift (tuple): Z/Y/X 三个轴的最大平移体素数（含）。

    Returns:
        img, label (np.ndarray, np.ndarray)
    """
    if np.random.rand() >= prob:
        return img, label

    z, h, w = img.shape
    min_dz, min_dy, min_dx = min_shift
    max_dz, max_dy, max_dx = max_shift

    min_dz = max(0, int(min_dz))
    min_dy = max(0, int(min_dy))
    min_dx = max(0, int(min_dx))
    max_dz = max(min_dz, int(max_dz))
    max_dy = max(min_dy, int(max_dy))
    max_dx = max(min_dx, int(max_dx))

    max_dz = min(max_dz, z - 1)
    max_dy = min(max_dy, h - 1)
    max_dx = min(max_dx, w - 1)

    if max_dz <= 0 and max_dy <= 0 and max_dx <= 0:
        return img, label

    min_dz = min(min_dz, max_dz)
    min_dy = min(min_dy, max_dy)
    min_dx = min(min_dx, max_dx)

    mag_dz = np.random.randint(min_dz, max_dz + 1) if max_dz > 0 else 0
    mag_dy = np.random.randint(min_dy, max_dy + 1) if max_dy > 0 else 0
    mag_dx = np.random.randint(min_dx, max_dx + 1) if max_dx > 0 else 0

    dz = mag_dz * np.random.choice((-1, 1)) if mag_dz > 0 else 0
    dy = mag_dy * np.random.choice((-1, 1)) if mag_dy > 0 else 0
    dx = mag_dx * np.random.choice((-1, 1)) if mag_dx > 0 else 0

    img = _shift_with_padding(img, dz, dy, dx)
    label = _shift_with_padding(label, dz, dy, dx)
    return img, label