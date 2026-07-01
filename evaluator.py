import os

import numpy as np
import tifffile as tiff
import torch
import wandb
import matplotlib.pyplot as plt

from training.axis_utils import normalize_to_zyx
from models import UNet3D
from config.val_config import get_validation_config
from validate.metrics import dice_coefficient, iou_score, precision_recall_f1_specificity
from evaluation import (
    save_validation_visualization,
    load_validation_pairs,
    save_prediction_results,
)
from evaluation.inference import sliding_window_inference
from evaluation.loss_factory import build_validation_criterion
from evaluation.pr_curve import sample_for_curves
from utils import (
    log_pr_roc_to_wandb,
    log_f1_curve_to_wandb,
    log_sample_table_to_wandb,
    log_summary_table_to_wandb,
)


VAL_CONFIG = get_validation_config()


def evaluate_model(
    model_path,
    val_img_dir,
    val_label_dir,
    patch_size=VAL_CONFIG["patch_size"],
    stride=VAL_CONFIG["stride"],
    threshold=VAL_CONFIG["threshold"],
    dust_remove_min_size=VAL_CONFIG["dust_remove_min_size"],
    loss_type=VAL_CONFIG["loss_type"],
    save_results=VAL_CONFIG["save_results"],
    wandb_run=None,
):
    # Set device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Optionally build criterion if loss logging is enabled
    criterion = build_validation_criterion(loss_type)

    # Load validation pairs
    pairs = load_validation_pairs(val_img_dir, val_label_dir)
    if not pairs:
        raise ValueError(
            f"No validation image/label pairs found in '{val_img_dir}' and '{val_label_dir}'."
        )

    # Initialize lists to collect metrics and curve data across samples
    dice_list = []
    iou_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    accuracy_list = []
    loss_list = []
    sample_rows = []
    curve_true = []
    curve_score = []

    # Iterate through validation samples
    with torch.no_grad():
        for i, (img_path, label_path) in enumerate(pairs, start=1):
            sample_name = os.path.basename(img_path)
            vol = tiff.imread(img_path).astype(np.float32)
            lab = tiff.imread(label_path).astype(np.float32)

            if vol.ndim != 3 or lab.ndim != 3:
                raise ValueError(
                    f"Expected 3D volumes for '{sample_name}', got image ndim={vol.ndim}, label ndim={lab.ndim}."
                )

            vol, _ = normalize_to_zyx(vol, img_path, patch_size)
            lab, _ = normalize_to_zyx(lab, label_path, patch_size)

            if vol.shape != lab.shape:
                raise ValueError(
                    f"Shape mismatch after transpose for '{sample_name}': image {vol.shape}, label {lab.shape}."
                )

            prob_map, pred_seg, sample_loss = sliding_window_inference(
                vol,
                lab,
                model,
                patch_size=patch_size,
                stride=stride,
                threshold=threshold,
                dust_remove_min_size=dust_remove_min_size,
                device=device,
                criterion=criterion,
            )

            gt_seg = (lab > 0).astype(np.uint8)

            dice = dice_coefficient(pred_seg, gt_seg)
            iou = iou_score(pred_seg, gt_seg)

            precision, recall, f1, specificity, accuracy = precision_recall_f1_specificity(pred_seg, gt_seg)

            dice_list.append(dice)
            iou_list.append(iou)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            specificity_list.append(specificity)
            accuracy_list.append(accuracy)

            if sample_loss is not None:
                loss_list.append(sample_loss)

            if save_results:
                save_prediction_results(prob_map, pred_seg, img_path, out_dir="validation_results")


            # plot and log PR/ROC curve data for this sample
            y_true, y_score = sample_for_curves(gt_seg, prob_map)
            curve_true.append(y_true)
            curve_score.append(y_score)

            # Build per-sample visualization and upload directly to wandb (no file saved)
            sample_image = None
            if wandb_run is not None:
                fig = save_validation_visualization(
                    volume=vol,
                    label=lab,
                    pred_seg=pred_seg,
                    prob_map=prob_map,
                )
                sample_image = wandb.Image(fig, caption=f"{sample_name} | val visualization")
                plt.close(fig)

            sample_metrics = {
                "dice": float(dice),
                "iou": float(iou),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "accuracy": float(accuracy),
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
                    "accuracy": sample_metrics["accuracy"],
                    "loss": sample_metrics["loss"],
                    "sample_image": sample_image,
                }
            )

    # Plot and log PR/ROC curves if any curve data was collected
    if curve_true and wandb_run is not None:
        y_true_all = np.concatenate(curve_true)
        y_score_all = np.concatenate(curve_score)
        log_pr_roc_to_wandb(wandb_run, y_true_all, y_score_all)
        log_f1_curve_to_wandb(wandb_run, y_true_all, y_score_all)

    # mean metrics summary and logging
    summary = {
        "dice": float(np.mean(dice_list)),
        "iou": float(np.mean(iou_list)),
        "f1": float(np.mean(f1_list)),
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "specificity": float(np.mean(specificity_list)),
        "accuracy": float(np.mean(accuracy_list)),
    }
    if loss_list:
        summary["loss"] = float(np.mean(loss_list))

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
    dust_remove_min_size = config["dust_remove_min_size"]
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
                "dust_remove_min_size": dust_remove_min_size,
                "loss_type": loss_type,
                "save_results": save_results,
            },
            job_type="validation",
            # settings=wandb.Settings(silent=True, console="off"),
        )

    try:
        summary = evaluate_model(
            model_path=model_path,
            val_img_dir=val_img_dir,
            val_label_dir=val_label_dir,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            dust_remove_min_size=dust_remove_min_size,
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
        print(f"Accuracy: {summary['accuracy']:.4f}")
        if "loss" in summary:
            print(f"Validation Loss: {summary['loss']:.4f}")

    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
