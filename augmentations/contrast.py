import numpy as np


def random_contrast_3d(
    img,
    label,
    prob=0.3,
    contrast_range=(0.5, 1.5),
    brightness_range=(-0.25, 0.25),
    gamma_log2_range=(-1.0, 1.0),
):
    print("🔄 Applying random_contrast_3d augmentation...")
    """随机对比度 + 亮度偏移 + gamma 变换。"""
    if np.random.rand() < prob:
        a = np.random.uniform(contrast_range[0], contrast_range[1])
        b = np.random.uniform(brightness_range[0], brightness_range[1])
        img = img * a + b

        gamma = 2 ** np.random.uniform(gamma_log2_range[0], gamma_log2_range[1])
        v_min, v_max = img.min(), img.max()
        if v_max > v_min:
            img_01 = (img - v_min) / (v_max - v_min)
            img_01 = np.clip(img_01, 0.0, 1.0)
            img = (img_01 ** gamma) * (v_max - v_min) + v_min

    return img, label