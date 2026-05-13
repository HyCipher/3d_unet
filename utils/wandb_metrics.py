import wandb


# ── Training Logging ───────────────────────────────────────────────────────────

def log_training_loss(epoch, train_loss):
    """Log train loss for one epoch."""
    wandb.log({"train_loss": train_loss, "epoch": epoch})


# ── Validation Metrics ─────────────────────────────────────────────────────────

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
