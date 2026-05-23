import numpy as np


def random_missing_section(img, label, prob=0.3, max_missing=2):
    """
    模拟串行切片 EM 中的缺失切片伪影。
    """
    if np.random.rand() >= prob:
        return img, label

    z_dim = img.shape[0]
    if z_dim < 4:
        return img, label

    n_missing = np.random.randint(1, max_missing + 1)
    # 避免选到首尾切片（边界切片缺失对网络影响过大）
    candidates = np.arange(1, z_dim - 1)
    chosen = np.random.choice(candidates, size=min(n_missing, len(candidates)), replace=False)

    img = img.copy()
    for z in chosen:
        img[z] = 0.0  # 以 0 填充（对应 z-score 后的均值附近）

    return img, label
