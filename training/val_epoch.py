import glob
import os
import numpy as np
import tifffile as tiff
from .axis_utils import normalize_to_zyx
from evaluation.inference import sliding_window_inference
from validate.metrics import dice_coefficient


def val_one_epoch(model, controls, device):
    """Sliding-window validation over all val volumes. Returns mean Dice."""
    img_paths = sorted(glob.glob(os.path.join(controls["val_img_dir"], "*.tif")))
    label_paths = sorted(glob.glob(os.path.join(controls["val_label_dir"], "*.tif")))

    if not img_paths:
        print("[val] No validation images found — skipping.")
        return 0.0

    dice_scores = []
    for img_path, lbl_path in zip(img_paths, label_paths):
        vol, _ = normalize_to_zyx(tiff.imread(img_path).astype(np.float32), img_path)
        lab, _ = normalize_to_zyx(tiff.imread(lbl_path).astype(np.float32), lbl_path)

        _, pred_seg, _ = sliding_window_inference(
            vol, lab, model,
            patch_size=tuple(controls["val_patch_size"]),
            stride=tuple(controls["val_stride"]),
            threshold=float(controls.get("val_threshold", 0.5)),
            dust_remove_min_size=int(controls.get("dust_remove_min_size", 0)),
            device=device,
        )
        dice_scores.append(dice_coefficient(pred_seg, (lab > 0).astype(np.uint8)))

    mean_dice = float(np.mean(dice_scores))
    print(f"[val] mean Dice = {mean_dice:.4f}  ({len(dice_scores)} volumes)")
    return mean_dice
