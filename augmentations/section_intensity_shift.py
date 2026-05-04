import numpy as np


def random_section_intensity_shift(img, label, prob=0.4, std=0.1):
    """模拟逐切片染色/焦距漂移伪影。

    在 serial-section EM 中，每张切片的染色程度和电子束焦距可能略有不同，
    导致相邻切片之间存在全局亮度偏移。此增强为每个 Z 切片独立施加随机的
    亮度偏移（加法）和对比度缩放（乘法），模拟真实采集条件。

    Args:
        img   (np.ndarray): 图像数组，shape 为 (Z, Y, X)，z-scored 归一化后。
        label (np.ndarray): 标签数组，shape 为 (Z, Y, X)，不做修改。
        prob  (float): 触发概率。
        std   (float): 亮度偏移的高斯标准差（相对于归一化后的数值范围）。

    Returns:
        img, label (np.ndarray, np.ndarray)
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
