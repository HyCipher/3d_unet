import os
import glob
import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_erosion


def validate_hard_negative_setup(img_paths, hard_negative_dir, hard_negative_sample_ratio):
    """Validate hard-negative configuration against dataset inputs."""
    if hard_negative_dir is not None:
        image_basenames = {os.path.basename(path) for path in img_paths}
        hard_negative_basenames = {
            os.path.basename(path)
            for path in glob.glob(os.path.join(hard_negative_dir, "*.tif"))
        }

        missing_masks = sorted(image_basenames - hard_negative_basenames)
        extra_masks = sorted(hard_negative_basenames - image_basenames)
        if missing_masks or extra_masks:
            missing_preview = missing_masks[:5]
            extra_preview = extra_masks[:5]
            raise ValueError(
                "Hard negative mask basenames must exactly match training image basenames. "
                f"Missing ({len(missing_masks)}): {missing_preview}; "
                f"Extra ({len(extra_masks)}): {extra_preview}"
            )
    elif hard_negative_sample_ratio > 0.0:
        raise ValueError("hard_negative_sample_ratio > 0 requires hard_negative_dir.")


def build_hard_negative_coordinates(
    img_paths,
    hard_negative_dir,
    patch_size,
    normalize_to_zyx_fn,
    hardest_negative=False,
    hardest_negative_erosion_iters=1,
):
    """Load hard-negative masks and cache coordinate arrays for sampling."""
    hard_negative_coords = []
    hardest_negative_coords = []

    if hard_negative_dir is None:
        return hard_negative_coords, hardest_negative_coords

    for img_path in img_paths:
        mask_path = os.path.join(hard_negative_dir, os.path.basename(img_path))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing hard negative mask: {mask_path}")

        hard_negative_mask = tiff.imread(mask_path).astype(np.float32)
        hard_negative_mask, _ = normalize_to_zyx_fn(hard_negative_mask, mask_path, patch_size)
        hard_negative_mask = hard_negative_mask > 0
        hard_negative_coords.append(np.argwhere(hard_negative_mask))

        if hardest_negative:
            hardest_mask = binary_erosion(
                hard_negative_mask,
                iterations=max(int(hardest_negative_erosion_iters), 1),
                border_value=0,
            )
            if hardest_mask.any():
                hardest_negative_coords.append(np.argwhere(hardest_mask))
            else:
                hardest_negative_coords.append(np.argwhere(hard_negative_mask))

    return hard_negative_coords, hardest_negative_coords


def choose_hard_negative_coords(vid, hard_negative_coords, hardest_negative_coords, hardest_negative=False):
    """Return the coordinate cache used for hard-negative sampling for one volume."""
    if hardest_negative and hardest_negative_coords:
        return hardest_negative_coords[vid]
    return hard_negative_coords[vid]


def active_hard_negative_coords(hard_negative_coords, hardest_negative_coords, hardest_negative=False):
    """Return the active coordinate cache for logging/debug purposes."""
    if hardest_negative and hardest_negative_coords:
        return hardest_negative_coords
    return hard_negative_coords
