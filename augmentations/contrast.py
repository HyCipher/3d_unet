import numpy as np


def random_contrast_3d(
    img,
    label,
    prob=0.3,
    contrast_range=(0.5, 1.5),
    brightness_range=(-0.25, 0.25),
    gamma_log2_range=(-1.0, 1.0),
):
    """
        Randomly adjust contrast, brightness, and gamma of a 3D image.
        
        Args:
            img: 3D input image (D, H, W)
            label: 3D label (D, H, W), unchanged
            prob: probability of applying this augmentation
            contrast_range: range for random contrast scaling factor
            brightness_range: range for random brightness offset
            gamma_log2_range: range for random gamma in log2 space
        Returns:
            Augmented image and original label
        """
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