from datetime import datetime

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


def log_train_loss(epoch, train_loss):
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
