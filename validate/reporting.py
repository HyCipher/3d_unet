def print_metrics(train_metrics, val_metrics):
    """Console-friendly metrics output."""
    if train_metrics is not None:
        print(
            "Train Metrics -> "
            f"Dice: {train_metrics['dice']:.4f}, "
            f"IoU: {train_metrics['iou']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}, "
            f"Precision: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, "
            f"Specificity: {train_metrics['specificity']:.4f}"
        )

    if "loss" in val_metrics:
        print(f"Validation Loss: {val_metrics['loss']:.4f}")

    print(
        "Validation Metrics -> "
        f"Dice: {val_metrics['dice']:.4f}, "
        f"IoU: {val_metrics['iou']:.4f}, "
        f"F1: {val_metrics['f1']:.4f}, "
        f"Precision: {val_metrics['precision']:.4f}, "
        f"Recall: {val_metrics['recall']:.4f}, "
        f"Specificity: {val_metrics['specificity']:.4f}"
    )