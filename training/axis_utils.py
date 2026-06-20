import numpy as np


def normalize_to_zyx(volume, volume_path="<array>", patch_size=None):
    """Normalize a 3D volume to (Z, H, W) using a conservative axis heuristic."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape} for {volume_path}")

    if volume.shape[0] < volume.shape[1] and volume.shape[0] < volume.shape[2]:
        zyx = volume
        source_order = "ZYX"
    elif volume.shape[1] < volume.shape[0] and volume.shape[1] < volume.shape[2]:
        zyx = np.transpose(volume, (1, 0, 2))
        source_order = "YZX"
    elif volume.shape[2] < volume.shape[0] and volume.shape[2] < volume.shape[1]:
        zyx = np.transpose(volume, (2, 0, 1))
        source_order = "YXZ"
    else:
        raise ValueError(
            "Cannot reliably infer Z axis from shape "
            f"{volume.shape} for {volume_path}. "
            "Please convert volume to (Z,H,W) explicitly."
        )

    if patch_size is not None:
        pd, ph, pw = patch_size
        zd, hd, wd = zyx.shape
        if pd > zd or ph > hd or pw > wd:
            raise ValueError(
                f"Patch size {patch_size} does not fit volume {zyx.shape} for {volume_path}"
            )

    return zyx, source_order


def zyx_to_hwz(volume, volume_path="<array>"):
    """Convert a (Z, H, W) volume back to (H, W, Z)."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape} for {volume_path}")
    return np.transpose(volume, (1, 2, 0))