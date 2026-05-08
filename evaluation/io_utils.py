import glob
import os

import numpy as np
import tifffile as tiff


def load_validation_pairs(img_dir, label_dir, ext="*.tif"):
    """Load image/label path pairs with basic validation checks."""
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    img_paths = sorted(glob.glob(os.path.join(img_dir, ext)))
    label_paths = sorted(glob.glob(os.path.join(label_dir, ext)))

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No {ext} files found in {img_dir}")
    if len(label_paths) == 0:
        raise FileNotFoundError(f"No {ext} files found in {label_dir}")
    if len(img_paths) != len(label_paths):
        raise ValueError(
            f"Image/label count mismatch: {len(img_paths)} images vs {len(label_paths)} labels"
        )

    return list(zip(img_paths, label_paths))


def save_prediction_results(prob_map, pred_seg, img_path, out_dir="validation_results"):
    """Persist prediction and probability maps as TIFF files."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(img_path)

    # Save as (H, W, Z) for consistency with source tif layout.
    pred_hwz = np.transpose(pred_seg.astype(np.uint8) * 255, (1, 2, 0))
    prob_hwz = np.transpose(prob_map.astype(np.float32), (1, 2, 0))

    pred_path = os.path.join(out_dir, f"pred_{base}")
    prob_path = os.path.join(out_dir, f"prob_{base}")
    tiff.imwrite(pred_path, pred_hwz)
    tiff.imwrite(prob_path, prob_hwz)
