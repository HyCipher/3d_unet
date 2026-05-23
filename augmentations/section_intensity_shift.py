import numpy as np


def random_section_intensity_shift(img, label, prob=0.4, std=0.1):
    """
    模拟逐切片染色/焦距漂移伪影。
    """
    if np.random.rand() >= prob:
        return img, label

    z_dim = img.shape[0]
    img = img.copy()

    # 逐切片独立偏移：亮度 + 轻微对比度扰动
    brightness_shifts = np.random.normal(0.0, std, size=z_dim).astype(np.float32)
    contrast_scales = np.random.normal(1.0, std * 0.3, size=z_dim).astype(np.float32)
    contrast_scales = np.clip(contrast_scales, 0.7, 1.3)

    for z in range(z_dim):
        img[z] = img[z] * contrast_scales[z] + brightness_shifts[z]

    return img, label
