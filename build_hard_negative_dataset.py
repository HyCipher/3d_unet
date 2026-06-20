import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_erosion
from training.axis_utils import normalize_to_zyx, zyx_to_hwz


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_tif(path: Path) -> np.ndarray:
    return tiff.imread(path)


def edge_mask_from_gt(gt_zyx: np.ndarray) -> np.ndarray:
    pos_mask = gt_zyx > 0
    padded = np.pad(pos_mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    interior = (
        center
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return center & (~interior)


def build_fp_mask(
    prob_map_zyx: np.ndarray,
    gt_zyx: np.ndarray,
    threshold: float,
    hardest_negative: bool,
    erosion_iters: int,
) -> np.ndarray:
    fp_mask = (prob_map_zyx > threshold) & (gt_zyx == 0)
    if hardest_negative:
        core_mask = binary_erosion(fp_mask, iterations=max(erosion_iters, 1), border_value=0)
        if core_mask.any():
            fp_mask = core_mask
    return fp_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a hard-negative dataset from saved probability maps.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing saved probability maps (*.tif).")
    parser.add_argument("--train-img-dir", required=True, help="Directory containing training images (*.tif).")
    parser.add_argument("--train-label-dir", required=True, help="Directory containing training labels (*.tif).")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Probability threshold for hard negatives.")
    parser.add_argument(
        "--hardest-negative",
        action="store_true",
        help="Use an eroded subset of the false-positive mask for stricter sampling.",
    )
    parser.add_argument(
        "--erosion-iters",
        type=int,
        default=1,
        help="Number of binary erosion iterations used when --hardest-negative is enabled.",
    )
    parser.add_argument("--enable-hard-negative", action="store_true", help="Write hard_negative_masks and config metadata.")
    parser.add_argument("--pos-ratio", type=float, default=0.4, help="Sampling ratio for positive-centered patches.")
    parser.add_argument("--edge-ratio", type=float, default=0.1, help="Sampling ratio for edge-centered patches.")
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.3,
        help="Sampling ratio for hard-negative-centered patches.",
    )
    parser.add_argument(
        "--pred-prefix",
        type=str,
        default="prob_",
        help="Prefix to strip from prediction filenames to get image filenames.",
    )
    parser.add_argument(
        "--img-token",
        type=str,
        default="img",
        help="Token in image filename that maps to label token.",
    )
    parser.add_argument(
        "--label-token",
        type=str,
        default="syn",
        help="Token used in label filenames corresponding to image token.",
    )
    return parser.parse_args()


def derive_image_name(pred_name: str, pred_prefix: str) -> str:
    if pred_prefix and pred_name.startswith(pred_prefix):
        return pred_name[len(pred_prefix) :]
    return pred_name


def derive_label_name(image_name: str, img_token: str, label_token: str) -> str:
    # Prefer replacing token surrounded by underscores to avoid accidental matches.
    surrounded = f"_{img_token}_"
    if surrounded in image_name:
        return image_name.replace(surrounded, f"_{label_token}_", 1)

    # Fallback: replace first plain token occurrence.
    if img_token in image_name:
        return image_name.replace(img_token, label_token, 1)
    return image_name


def main() -> None:
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    train_img_dir = Path(args.train_img_dir)
    train_label_dir = Path(args.train_label_dir)
    output_dir = Path(args.output_dir)

    pred_paths = sorted(pred_dir.glob("*.tif"))
    if not pred_paths:
        raise FileNotFoundError(f"No .tif probability maps found in {pred_dir}")

    img_lookup = {p.name: p for p in train_img_dir.glob("*.tif")}
    label_lookup = {p.name: p for p in train_label_dir.glob("*.tif")}

    if not img_lookup:
        raise FileNotFoundError(f"No .tif training images found in {train_img_dir}")
    if not label_lookup:
        raise FileNotFoundError(f"No .tif training labels found in {train_label_dir}")

    image_out = output_dir / "images"
    label_out = output_dir / "labels"
    mask_out = output_dir / "hard_negative_masks"
    ensure_dir(image_out)
    ensure_dir(label_out)
    if args.enable_hard_negative:
        ensure_dir(mask_out)

    total_pos_voxels = 0
    total_edge_voxels = 0
    total_hard_negative_voxels = 0
    total_random_ratio = max(0.0, 1.0 - args.pos_ratio - args.edge_ratio - args.hard_negative_ratio)
    per_volume_stats = []

    for pred_path in pred_paths:
        pred_name = pred_path.name
        image_name = derive_image_name(pred_name, args.pred_prefix)
        label_name = derive_label_name(image_name, args.img_token, args.label_token)

        if image_name not in img_lookup:
            raise FileNotFoundError(
                f"Missing matching image for pred={pred_name}. Expected image={image_name}"
            )
        if label_name not in label_lookup:
            raise FileNotFoundError(
                f"Missing matching label for pred={pred_name}. Expected label={label_name}"
            )

        prob_map = load_tif(pred_path)
        label = load_tif(label_lookup[label_name])

        if prob_map.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for pred={pred_name}: prob={prob_map.shape}, label={label.shape}"
            )

        prob_map_zyx, _ = normalize_to_zyx(prob_map, str(pred_path))
        prob_map_zyx = prob_map_zyx.astype(np.float32)
        gt_zyx, _ = normalize_to_zyx(label, str(label_lookup[label_name]))
        gt_zyx = gt_zyx.astype(np.float32)
        pos_mask = gt_zyx > 0
        edge_mask = edge_mask_from_gt(gt_zyx)
        fp_mask = build_fp_mask(
            prob_map_zyx=prob_map_zyx,
            gt_zyx=gt_zyx,
            threshold=args.threshold,
            hardest_negative=args.hardest_negative,
            erosion_iters=args.erosion_iters,
        )

        pos_voxels = int(pos_mask.sum())
        edge_voxels = int(edge_mask.sum())
        hard_negative_voxels = int(fp_mask.sum())
        random_voxels = int(gt_zyx.size - hard_negative_voxels - pos_voxels)

        total_pos_voxels += pos_voxels
        total_edge_voxels += edge_voxels
        total_hard_negative_voxels += hard_negative_voxels

        link_or_copy(img_lookup[image_name], image_out / image_name)
        link_or_copy(label_lookup[label_name], label_out / label_name)

        if args.enable_hard_negative:
            tiff.imwrite(mask_out / image_name, zyx_to_hwz(fp_mask.astype(np.uint8) * 255, image_name))

        per_volume_stats.append(
            {
                "pred_name": pred_name,
                "image_name": image_name,
                "label_name": label_name,
                "pos_voxels": pos_voxels,
                "edge_voxels": edge_voxels,
                "hard_negative_voxels": hard_negative_voxels,
                "random_voxels": random_voxels,
            }
        )

        print(
            f"[hard_negative] pred={pred_name} -> image={image_name}, label={label_name}: "
            f"pos={pos_voxels}, edge={edge_voxels}, "
            f"hard_negative={hard_negative_voxels}, random={random_voxels}"
        )

    config = {
        "enabled": bool(args.enable_hard_negative),
        "threshold": float(args.threshold),
        "hardest_negative": bool(args.hardest_negative),
        "erosion_iters": int(args.erosion_iters),
        "pos_sample_ratio": float(args.pos_ratio),
        "edge_sample_ratio": float(args.edge_ratio),
        "hard_negative_sample_ratio": float(args.hard_negative_ratio),
        "random_sample_ratio": float(total_random_ratio),
        "total_pos_voxels": int(total_pos_voxels),
        "total_edge_voxels": int(total_edge_voxels),
        "total_hard_negative_voxels": int(total_hard_negative_voxels),
        "volumes": per_volume_stats,
        "source_pred_dir": str(pred_dir),
        "source_train_img_dir": str(train_img_dir),
        "source_train_label_dir": str(train_label_dir),
    }

    ensure_dir(output_dir)
    with open(output_dir / "hard_negative_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(
        f"Done. output_dir={output_dir}, enabled={config['enabled']}, "
        f"pos_ratio={config['pos_sample_ratio']}, edge_ratio={config['edge_sample_ratio']}, "
        f"hard_negative_ratio={config['hard_negative_sample_ratio']}, random_ratio={config['random_sample_ratio']}"
    )


if __name__ == "__main__":
    main()