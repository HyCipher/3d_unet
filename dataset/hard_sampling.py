import os
import glob
import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_erosion
from scipy import ndimage


def validate_hard_negative_setup(
    img_paths,
    hard_negative_dir,
    hard_negative_sample_ratio,
    hard_positive_sample_ratio=0.0,
):
    """Validate hard-sampling configuration against dataset inputs."""
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
    elif hard_negative_sample_ratio > 0.0 or hard_positive_sample_ratio > 0.0:
        raise ValueError(
            "hard_negative_sample_ratio > 0 or hard_positive_sample_ratio > 0 requires hard_negative_dir."
        )


def _keep_pred_components_without_gt_overlap(pred_mask, gt_mask, connectivity=1):
    """Keep predicted connected components only when they do not overlap GT at all."""
    structure = ndimage.generate_binary_structure(pred_mask.ndim, connectivity)
    labeled_pred, n_pred = ndimage.label(pred_mask, structure=structure)
    if n_pred == 0:
        return np.zeros_like(pred_mask, dtype=bool)

    overlap_counts = np.bincount(labeled_pred[gt_mask], minlength=n_pred + 1)
    overlaps_gt = overlap_counts > 0

    keep_labels = np.ones(n_pred + 1, dtype=bool)
    keep_labels[0] = False
    keep_labels[overlaps_gt] = False
    return keep_labels[labeled_pred]


def _keep_gt_components_without_pred_overlap(gt_mask, pred_mask, connectivity=1):
    """Keep GT connected components only when they have no overlap with any prediction."""
    structure = ndimage.generate_binary_structure(gt_mask.ndim, connectivity)
    labeled_gt, n_gt = ndimage.label(gt_mask, structure=structure)
    if n_gt == 0:
        return np.zeros_like(gt_mask, dtype=bool)

    overlap_counts = np.bincount(labeled_gt[pred_mask], minlength=n_gt + 1)
    overlaps_pred = overlap_counts > 0

    keep_labels = np.ones(n_gt + 1, dtype=bool)
    keep_labels[0] = False
    keep_labels[overlaps_pred] = False
    return keep_labels[labeled_gt]


def _load_pred_mask(mask_path, patch_size, normalize_to_zyx_fn):
    pred_mask = tiff.imread(mask_path).astype(np.float32)
    pred_mask, _ = normalize_to_zyx_fn(pred_mask, mask_path, patch_size)
    return pred_mask > 0


def build_hard_negative_coordinates(
    img_paths,
    labels,
    hard_negative_dir,
    patch_size,
    normalize_to_zyx_fn,
    hardest_negative=False,
    hardest_negative_erosion_iters=1,
):
    """Build hard-negative coordinates from predicted components without GT overlap."""
    hard_negative_coords = []
    hardest_negative_coords = []

    if hard_negative_dir is None:
        return hard_negative_coords, hardest_negative_coords

    for vid, img_path in enumerate(img_paths):
        mask_path = os.path.join(hard_negative_dir, os.path.basename(img_path))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing hard negative mask: {mask_path}")

        pred_mask = _load_pred_mask(mask_path, patch_size, normalize_to_zyx_fn)
        gt_mask = labels[vid] > 0

        hard_negative_mask = _keep_pred_components_without_gt_overlap(pred_mask, gt_mask)
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


def build_hard_positive_coordinates(
    img_paths,
    labels,
    hard_negative_dir,
    patch_size,
    normalize_to_zyx_fn,
):
    """Build hard-positive coordinates from GT components without prediction overlap."""
    hard_positive_coords = []

    if hard_negative_dir is None:
        return hard_positive_coords

    for vid, img_path in enumerate(img_paths):
        mask_path = os.path.join(hard_negative_dir, os.path.basename(img_path))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing hard negative mask: {mask_path}")

        pred_mask = _load_pred_mask(mask_path, patch_size, normalize_to_zyx_fn)
        gt_mask = labels[vid] > 0
        hard_positive_mask = _keep_gt_components_without_pred_overlap(gt_mask, pred_mask)
        hard_positive_coords.append(np.argwhere(hard_positive_mask))

    return hard_positive_coords


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