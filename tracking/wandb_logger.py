from datetime import datetime

import numpy as np
import wandb


def build_wandb_config(loader, lr, controls):
    """Build wandb config from runtime values."""
    config = {
        "architecture": "3D UNet",
        "epochs": controls["num_epochs"],
        "batch_size": loader.batch_size,
        "learning_rate": lr,
        "patch_size": loader.dataset.patch_size,
        "val_patch_size": controls["val_patch_size"],
        "val_stride": controls["val_stride"],
        "val_threshold": controls["val_threshold"],
        "loss_function": controls["loss_type"],
        "validate_every": controls["validate_every"],
        "eval_train_set": controls["eval_train_set"],
    }
    if controls["loss_type"] == "dicefocal":
        config["dice_weight"] = controls["dice_weight"]
        config["focal_weight"] = controls["focal_weight"]
    return config


def init_wandb_run(project, config):
    """Initialize a wandb run with a timestamped name."""
    wandb.init(
        project=project,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
    )


def log_training_loss(epoch, train_loss):
    """Log train loss for one epoch."""
    wandb.log({"train_loss": train_loss, "epoch": epoch})


def log_validation_to_wandb(train_metrics, val_metrics, epoch):
    """Send validation metrics to wandb."""
    train_metrics = train_metrics or {}
    payload = {
        "epoch": epoch,
        "train_dice": train_metrics.get("dice"),
        "train_iou": train_metrics.get("iou"),
        "train_f1": train_metrics.get("f1"),
        "train_precision": train_metrics.get("precision"),
        "train_recall": train_metrics.get("recall"),
        "train_specificity": train_metrics.get("specificity"),
        "val_dice": val_metrics["dice"],
        "val_iou": val_metrics["iou"],
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_specificity": val_metrics["specificity"],
    }
    if "loss" in val_metrics:
        payload["val_loss"] = val_metrics["loss"]
    wandb.log(payload)


def finish_wandb_run():
    """Close current wandb run."""
    wandb.finish()


def log_pr_roc_to_wandb(wandb_run, y_true, y_score):
    """Log PR/ROC curves to wandb using native curve visualizations."""
    if wandb_run is None:
        print("Skip PR/ROC upload: wandb is disabled.")
        return

    if y_true.size == 0:
        print("Skip PR/ROC plot: no sampled points.")
        return

    if np.unique(y_true).size < 2:
        print("Skip PR/ROC plot: ground truth has only one class.")
        return
    
    y_true = y_true.astype(np.int32)
    y_score = np.clip(y_score.astype(np.float32), 0.0, 1.0)
    y_proba = np.stack([1.0 - y_score, y_score], axis=1)

    wandb_run.log(
        {
            "validation/pr_curve": wandb.plot.pr_curve(
                y_true,
                y_proba,
                labels=["background", "foreground"],
            ),
            "validation/roc_curve": wandb.plot.roc_curve(
                y_true,
                y_proba,
                labels=["background", "foreground"],
            ),
        }
    )
    print("PR/ROC curves logged to wandb.")


def build_center_slice_log(volume, label, pred_seg, prob_map, sample_name):
    """Build wandb images for the center slice of one 3D sample."""
    z_mid = volume.shape[0] // 2

    img_slice = volume[z_mid]
    gt_slice = (label[z_mid] > 0).astype(np.uint8)
    pred_slice = pred_seg[z_mid].astype(np.uint8)
    prob_slice = prob_map[z_mid].astype(np.float32)

    return {
        "validation/original_slice": wandb.Image(
            img_slice,
            caption=f"{sample_name} | center slice original",
        ),
        "validation/ground_truth_slice": wandb.Image(
            gt_slice,
            caption=f"{sample_name} | center slice label",
        ),
        "validation/prediction_slice": wandb.Image(
            pred_slice,
            caption=f"{sample_name} | center slice prediction",
        ),
        "validation/probability_slice": wandb.Image(
            prob_slice,
            caption=f"{sample_name} | center slice probability",
        ),
    }


def log_sample_to_wandb(wandb_run, sample_name, volume, label, pred_seg, prob_map, metrics, step):
    """Log representative slice images for one validation sample."""
    if wandb_run is None:
        return

    payload = {
        "validation/sample_step": step,
        "validation/sample_id": sample_name,
    }

    payload.update(build_center_slice_log(volume, label, pred_seg, prob_map, sample_name))
    wandb_run.log(payload)


def log_sample_table_to_wandb(wandb_run, sample_rows):
    """Upload per-sample metrics as a dedicated wandb table."""
    if wandb_run is None or not sample_rows:
        return

    columns = [
        "sample_index",
        "sample_name",
        "dice",
        "iou",
        "f1",
        "precision",
        "recall",
        "specificity",
        "loss",
    ]
    table = wandb.Table(columns=columns)
    for row in sample_rows:
        table.add_data(
            row.get("sample_index"),
            row.get("sample_name"),
            row.get("dice"),
            row.get("iou"),
            row.get("f1"),
            row.get("precision"),
            row.get("recall"),
            row.get("specificity"),
            row.get("loss"),
        )

    wandb_run.log({"validation/sample_table": table})


def log_generated_files_to_wandb(wandb_run, visualization_path=None):
    """Upload generated validation PNG files to wandb."""
    import os

    if wandb_run is None:
        return

    payload = {}
    if visualization_path and os.path.exists(visualization_path):
        payload["validation/summary_visualization"] = wandb.Image(visualization_path)

    if payload:
        wandb_run.log(payload)


def log_summary_table_to_wandb(wandb_run, summary):
    """Upload summary metrics as a wandb table."""
    if wandb_run is None or not summary:
        return

    table = wandb.Table(columns=["metric", "value"])
    for key, value in summary.items():
        table.add_data(key, float(value))

    wandb_run.log({"validation/summary_table": table})
