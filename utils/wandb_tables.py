import os
import wandb


# ── Validation Tables & Artifacts ─────────────────────────────────────────────

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
        "sample_image",
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
            row.get("sample_image"),
        )

    wandb_run.log({"val/sample_table": table})


def log_summary_table_to_wandb(wandb_run, summary):
    """Upload summary metrics as a wandb table."""
    if wandb_run is None or not summary:
        return

    table = wandb.Table(columns=["metric", "value"])
    for key, value in summary.items():
        table.add_data(key, float(value))

    wandb_run.log({"val/summary_table": table})


def log_generated_files_to_wandb(wandb_run, visualization_path=None):
    """Upload generated validation PNG files to wandb."""
    if wandb_run is None:
        return

    payload = {}
    if visualization_path and os.path.exists(visualization_path):
        payload["val/summary_visualization"] = wandb.Image(visualization_path)

    if payload:
        wandb_run.log(payload)
