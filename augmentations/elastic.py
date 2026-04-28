import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def _elastic_deform_inplane(arr, dy, dx, order):
    """Apply the same Y/X displacement field to each Z slice."""
    z, h, w = arr.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.array([yy + dy, xx + dx])

    out = np.empty_like(arr)
    for index in range(z):
        out[index] = map_coordinates(
            arr[index],
            coords,
            order=order,
            mode="reflect",
        )
    return out


def random_elastic_deformation_3d(
    img,
    label,
    prob=0.2,
    alpha=20.0,
    sigma=6.0,
):
    """随机弹性形变，沿 Y/X 平面共享位移场并保持 Z 维不变。"""
    if np.random.rand() >= prob:
        return img, label

    h, w = img.shape[1:]
    if h < 2 or w < 2:
        return img, label

    dy = gaussian_filter((np.random.rand(h, w) * 2.0 - 1.0), sigma=sigma, mode="reflect") * alpha
    dx = gaussian_filter((np.random.rand(h, w) * 2.0 - 1.0), sigma=sigma, mode="reflect") * alpha

    img = _elastic_deform_inplane(img, dy, dx, order=1)
    label = _elastic_deform_inplane(label, dy, dx, order=0)
    return img, label