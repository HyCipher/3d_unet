import numpy as np
import wandb


# ── Validation Curves ──────────────────────────────────────────────────────────

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
            "val/pr_curve": wandb.plot.pr_curve(
                y_true,
                y_proba,
                labels=["background", "foreground"],
            ),
            "val/roc_curve": wandb.plot.roc_curve(
                y_true,
                y_proba,
                labels=["background", "foreground"],
            ),
        }
    )
    print("PR/ROC curves logged to wandb.")


def log_f1_curve_to_wandb(wandb_run, y_true, y_score, num_thresholds=100):
    """Log F1 score vs threshold curve to wandb as a custom line plot."""
    if wandb_run is None:
        print("Skip F1 curve upload: wandb is disabled.")
        return

    if y_true.size == 0:
        print("Skip F1 curve: no sampled points.")
        return

    if np.unique(y_true).size < 2:
        print("Skip F1 curve: ground truth has only one class.")
        return

    y_true = y_true.astype(np.float32)
    y_score = np.clip(y_score.astype(np.float32), 0.0, 1.0)

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    f1_scores = []
    for t in thresholds:
        pred = (y_score >= t).astype(np.float32)
        tp = np.sum(pred * y_true)
        fp = np.sum(pred * (1.0 - y_true))
        fn = np.sum((1.0 - pred) * y_true)
        denom = 2.0 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        f1_scores.append(float(f1))

    table = wandb.Table(columns=["threshold", "f1"])
    for t, f1 in zip(thresholds.tolist(), f1_scores):
        table.add_data(round(t, 4), round(f1, 6))

    wandb_run.log(
        {
            "val/f1_curve": wandb.plot.line(
                table,
                x="threshold",
                y="f1",
                title="F1 Score vs Threshold",
            )
        }
    )
    print("F1 curve logged to wandb.")
