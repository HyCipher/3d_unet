import glob
import os

import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from nets.detect import UNet
from losses import FocalLoss, DiceFocalLoss
from val_config import get_validation_config
from validate.metrics import dice_coefficient, iou_score, precision_recall_f1_specificity
from tracking import (
    log_pr_roc_to_wandb,
    log_sample_table_to_wandb,
    log_summary_table_to_wandb,
)


VAL_CONFIG = get_validation_config()


def _gen_starts(length, patch, stride):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def sliding_window_inference(
    volume,
    label,
    model,
    patch_size=(16, 512, 512),
    stride=(8, 256, 256),
    threshold=0.5,
    device="cuda",
    criterion=None,
):
    model.eval()
    z_len, h_len, w_len = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = np.zeros((z_len, h_len, w_len), dtype=np.float32)
    count_map = np.zeros((z_len, h_len, w_len), dtype=np.float32)
    patch_losses = []

    z_starts = _gen_starts(z_len, pd, sd)
    y_starts = _gen_starts(h_len, ph, sh)
    x_starts = _gen_starts(w_len, pw, sw)

    with torch.no_grad():
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    patch = volume[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw].copy()
                    patch = (patch - patch.mean()) / (patch.std() + 1e-8)

                    xt = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    logits = model(xt)
                    probs = torch.sigmoid(logits)

                    output[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += probs.cpu().numpy()[0, 0]
                    count_map[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += 1.0

                    if criterion is not None and label is not None:
                        y_patch = label[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw].copy()
                        y_patch = (y_patch > 0).astype(np.float32)
                        yt = torch.from_numpy(y_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                        patch_losses.append(criterion(logits, yt).item())

    output /= np.maximum(count_map, 1e-8)
    pred_seg = (output > threshold).astype(np.uint8)
    avg_loss = float(np.mean(patch_losses)) if patch_losses else None
    return output, pred_seg, avg_loss


def load_validation_pairs(img_dir, label_dir):
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.tif")))

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No .tif files found in {img_dir}")
    if len(img_paths) != len(label_paths):
        raise ValueError(
            f"Image/label count mismatch: {len(img_paths)} images vs {len(label_paths)} labels"
        )

    return list(zip(img_paths, label_paths))


def save_prediction_results(prob_map, pred_seg, img_path, out_dir="validation_results"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(img_path)

    # Save as (H, W, Z) for consistency with source tif layout.
    pred_hwz = np.transpose(pred_seg.astype(np.uint8) * 255, (1, 2, 0))
    prob_hwz = np.transpose(prob_map.astype(np.float32), (1, 2, 0))

    pred_path = os.path.join(out_dir, f"pred_{base}")
    prob_path = os.path.join(out_dir, f"prob_{base}")
    tiff.imwrite(pred_path, pred_hwz)
    tiff.imwrite(prob_path, prob_hwz)


def sample_for_curves(gt_seg, prob_map, max_points=300000):
    y_true = gt_seg.reshape(-1).astype(np.uint8)
    y_score = prob_map.reshape(-1).astype(np.float32)

    if y_true.size > max_points:
        step = int(np.ceil(y_true.size / max_points))
        y_true = y_true[::step]
        y_score = y_score[::step]

    return y_true, y_score

# The following functions are adapted from the training script's wandb logging utilities, but modified for validation context and to avoid logging training-specific metrics.
def save_validation_visualization(
    volume,
    label,
    pred_seg,
    prob_map,
):
    # Use center slice for a quick visual sanity check.
    z_mid = volume.shape[0] // 2

    img_slice = volume[z_mid]
    gt_slice = (label[z_mid] > 0).astype(np.uint8)
    pred_slice = pred_seg[z_mid].astype(np.uint8)
    prob_slice = prob_map[z_mid]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(img_slice, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_slice, cmap="gray")
    axes[0, 1].set_title("Label")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_slice, cmap="gray")
    axes[0, 2].set_title("Prediction")
    axes[0, 2].axis("off")

    im = axes[1, 0].imshow(prob_slice, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Probability")
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(img_slice, cmap="gray")
    axes[1, 1].imshow(gt_slice, cmap="Reds", alpha=0.4)
    axes[1, 1].set_title("Ground Truth Overlay")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img_slice, cmap="gray")
    axes[1, 2].imshow(pred_slice, cmap="Blues", alpha=0.4)
    axes[1, 2].set_title("Prediction Overlay")
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


def evaluate_model(
    model_path,
    val_img_dir,
    val_label_dir,
    patch_size=VAL_CONFIG["patch_size"],
    stride=VAL_CONFIG["stride"],
    threshold=VAL_CONFIG["threshold"],
    loss_type=VAL_CONFIG["loss_type"],
    save_results=VAL_CONFIG["save_results"],
    wandb_run=None,
):
    # Set device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Optionally build criterion if loss logging is enabled
    criterion = None
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif loss_type == "dicefocal":
        criterion = DiceFocalLoss(alpha=0.25, gamma=2.0, dice_weight=0.8, focal_weight=1.5)

    # Load validation pairs
    pairs = load_validation_pairs(val_img_dir, val_label_dir)

    # Initialize lists to collect metrics and curve data across samples
    dice_list = []
    iou_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    loss_list = []
    sample_rows = []
    curve_true = []
    curve_score = []
    first_sample_image = None

    # Iterate through validation samples
    for i, (img_path, label_path) in enumerate(pairs, start=1):
        sample_name = os.path.basename(img_path)
        vol = tiff.imread(img_path).astype(np.float32)
        lab = tiff.imread(label_path).astype(np.float32)

        # (H, W, Z) -> (Z, H, W)
        vol = np.transpose(vol, (2, 0, 1))
        lab = np.transpose(lab, (2, 0, 1))

        prob_map, pred_seg, sample_loss = sliding_window_inference(
            vol,
            lab,
            model,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            device=device,
            criterion=criterion,
        )

        gt_seg = (lab > 0).astype(np.uint8)

        dice = dice_coefficient(pred_seg, gt_seg)
        iou = iou_score(pred_seg, gt_seg)

        precision, recall, f1, specificity = precision_recall_f1_specificity(pred_seg, gt_seg)

        dice_list.append(dice)
        iou_list.append(iou)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)

        if sample_loss is not None:
            loss_list.append(sample_loss)

        if save_results:
            save_prediction_results(prob_map, pred_seg, img_path, out_dir="validation_results")


        # plot and log PR/ROC curve data for this sample
        y_true, y_score = sample_for_curves(gt_seg, prob_map)
        curve_true.append(y_true)
        curve_score.append(y_score)

        # Build per-sample visualization and upload directly to wandb (no file saved)
        fig = save_validation_visualization(
            volume=vol,
            label=lab,
            pred_seg=pred_seg,
            prob_map=prob_map,
        )
        
        # Log the first sample's visualization as a representative summary image for the run
        sample_image = wandb.Image(fig, caption=f"{sample_name} | val visualization")
        plt.close(fig)
        if first_sample_image is None:
            first_sample_image = sample_image

        sample_metrics = {
            "dice": float(dice),
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "loss": float(sample_loss) if sample_loss is not None else None,
        }
        sample_rows.append(
            {
                "sample_index": i,
                "sample_name": sample_name,
                "dice": sample_metrics["dice"],
                "iou": sample_metrics["iou"],
                "f1": sample_metrics["f1"],
                "precision": sample_metrics["precision"],
                "recall": sample_metrics["recall"],
                "specificity": sample_metrics["specificity"],
                "loss": sample_metrics["loss"],
                "sample_image": sample_image,
            }
        )

    # Plot and log PR/ROC curves if any curve data was collected
    if curve_true:
        y_true_all = np.concatenate(curve_true)
        y_score_all = np.concatenate(curve_score)
        log_pr_roc_to_wandb(wandb_run, y_true_all, y_score_all)

    # mean metrics summary and logging
    summary = {
        "dice": float(np.mean(dice_list)),
        "iou": float(np.mean(iou_list)),
        "f1": float(np.mean(f1_list)),
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "specificity": float(np.mean(specificity_list)),
    }
    if loss_list:
        summary["loss"] = float(np.mean(loss_list))

    if wandb_run is not None and first_sample_image is not None:
        wandb_run.log({"val/summary_visualization": first_sample_image})

    if wandb_run is not None:
        log_sample_table_to_wandb(wandb_run, sample_rows)
        log_summary_table_to_wandb(wandb_run, summary)

    return summary


def main():
    config = VAL_CONFIG
    loss_type = config["loss_type"]
    model_path = config["model_path"]
    val_img_dir = config["val_img_dir"]
    val_label_dir = config["val_label_dir"]
    patch_size = tuple(config["patch_size"])
    stride = tuple(config["stride"])
    threshold = config["threshold"]
    save_results = config["save_results"]
    
    # wandb config
    use_wandb = config["wandb"]
    wandb_project = config["wandb_project"]
    wandb_run_name = config["wandb_run_name"]

    wandb_run = None
    if use_wandb:
        run_name = wandb_run_name
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                # "model_path": model_path,
                "val_img_dir": val_img_dir,
                "val_label_dir": val_label_dir,
                "patch_size": patch_size,
                "stride": stride,
                "threshold": threshold,
                "loss_type": loss_type,
                "save_results": save_results,
            },
            job_type="validation",
        )

    try:
        summary = evaluate_model(
            model_path=model_path,
            val_img_dir=val_img_dir,
            val_label_dir=val_label_dir,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            loss_type=loss_type,
            save_results=save_results,
            wandb_run=wandb_run,
        )

        print("\n=== Mean Metrics ===")
        print(f"Dice: {summary['dice']:.4f}")
        print(f"IoU: {summary['iou']:.4f}")
        print(f"F1: {summary['f1']:.4f}")
        print(f"Precision: {summary['precision']:.4f}")
        print(f"Recall: {summary['recall']:.4f}")
        print(f"Specificity: {summary['specificity']:.4f}")
        if "loss" in summary:
            print(f"Validation Loss: {summary['loss']:.4f}")

    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
