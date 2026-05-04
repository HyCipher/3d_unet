import numpy as np


def random_missing_section(img, label, prob=0.3, max_missing=2):
    """模拟串行切片 EM 中的缺失切片伪影。

    在 serial-section EM 采集过程中，切片可能因破损或粘连而缺失，
    导致 Z 轴上出现 1-2 张空白帧。此增强将随机 Z 切片置为零
    并保持对应标签不变（缺失切片无法标注，视为忽略区域）。

    Args:
        img         (np.ndarray): 图像数组，shape 为 (Z, Y, X)，z-scored 归一化后。
        label       (np.ndarray): 标签数组，shape 为 (Z, Y, X)，不做修改。
        prob        (float): 触发概率。
        max_missing (int): 最多缺失的切片数。

    Returns:
        img, label (np.ndarray, np.ndarray)
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
