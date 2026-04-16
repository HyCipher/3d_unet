import numpy as np
from .rotate import random_rotation_90_3d
from .flip import random_flip_3d
from .gaussian_noise import random_gaussian_noise
from .contrast import random_contrast_3d

# def random_flip_3d(img, label, prob=0.5):
#     """随机翻转（沿各个轴）。"""
#     if np.random.rand() < prob:
#         img = np.flip(img, axis=0).copy()
#         label = np.flip(label, axis=0).copy()

#     if np.random.rand() < prob:
#         img = np.flip(img, axis=1).copy()
#         label = np.flip(label, axis=1).copy()

#     if np.random.rand() < prob:
#         img = np.flip(img, axis=2).copy()
#         label = np.flip(label, axis=2).copy()

#     return img, label


# def random_rotation_90_3d(img, label, prob=0.5):
#     """随机90度旋转（XY平面）。"""
#     if np.random.rand() < prob:
#         k = np.random.randint(1, 4)
#         img = np.rot90(img, k, axes=(1, 2)).copy()
#         label = np.rot90(label, k, axes=(1, 2)).copy()

#     return img, label


# def random_gaussian_noise(img, label, prob=0.1, std=0.01):
#     """添加高斯噪声。"""
#     if np.random.rand() < prob:
#         noise = np.random.normal(0, std, img.shape)
#         img = img + noise

#     return img, label


# def random_contrast_3d(
#     img,
#     label,
#     prob=0.3,
#     contrast_range=(0.5, 1.5),
#     brightness_range=(-0.25, 0.25),
#     gamma_log2_range=(-1.0, 1.0),
# ):
#     """随机对比度 + 亮度偏移 + gamma 变换。"""
#     if np.random.rand() < prob:
#         a = np.random.uniform(contrast_range[0], contrast_range[1])
#         b = np.random.uniform(brightness_range[0], brightness_range[1])
#         img = img * a + b

#         gamma = 2 ** np.random.uniform(gamma_log2_range[0], gamma_log2_range[1])
#         v_min, v_max = img.min(), img.max()
#         if v_max > v_min:
#             img_01 = (img - v_min) / (v_max - v_min)
#             img_01 = np.clip(img_01, 0.0, 1.0)
#             img = (img_01 ** gamma) * (v_max - v_min) + v_min

#     return img, label


def apply_augmentation(img, label, augment=True):
    """统一的增强接口。"""
    if not augment:
        return img, label

    img, label = random_flip_3d(img, label, prob=0.8)
    img, label = random_rotation_90_3d(img, label, prob=0.8)
    img, label = random_contrast_3d(img, label, prob=0.5,
                                    contrast_range=(0.5, 1.5),brightness_range=(-0.25, 0.25), gamma_log2_range=(-1.0, 1.0)
                                    )
    img, label = random_gaussian_noise(img, label, prob=0.1, std=0.01)

    return img, label
