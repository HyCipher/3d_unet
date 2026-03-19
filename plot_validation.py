import json
import os

import matplotlib.pyplot as plt
import numpy as np


def has_valid_series(values):
    return values is not None and any(not np.isnan(value) for value in values)


def load_validation_history(history_path="validation_history.json"):
    """Load validation history in either legacy list format or dict-per-epoch format."""
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"{history_path} not found. Run training first to generate validation history."
        )

    with open(history_path, "r") as f:
        history = json.load(f)

    if not history:
        raise ValueError("validation history is empty")

    if isinstance(history[0], (int, float)):
        return {
            "epochs": list(range(10, len(history) * 10 + 1, 10)),
            "train": None,
            "validation": {"dice": history},
            "loss": None,
        }

    epochs = [item["epoch"] for item in history]
    metric_names = ["dice", "iou", "f1", "precision", "recall", "specificity"]
    has_train = any(isinstance(item.get("train"), dict) for item in history)

    train_metrics = None
    if has_train:
        train_metrics = {
            name: [item.get("train", {}).get(name, np.nan) if isinstance(item.get("train"), dict) else np.nan for item in history]
            for name in metric_names
        }

    val_metrics = {
        name: [item["validation"].get(name, np.nan) for item in history]
        for name in metric_names
    }

    # train_loss: training loss at each validation epoch
    # val_loss: validation dataset loss (may be absent in old files)
    train_losses = [item.get("train_loss", item.get("loss", np.nan)) for item in history]
    val_losses = [
        item.get("val_loss",
            item.get("validation", {}).get("loss", np.nan) if isinstance(item.get("validation"), dict) else np.nan
        )
        for item in history
    ]

    return {
        "epochs": epochs,
        "train": train_metrics,
        "validation": val_metrics,
        "loss": train_losses,
        "val_loss": val_losses,
    }
def plot_all_metrics_history(history_path="validation_history.json", output_path="validation_all_metrics_curve.png"):
    """Plot all validation metrics (and training metrics when available) in subplots."""
    history = load_validation_history(history_path)
    epochs = history["epochs"]
    train_metrics = history["train"]
    val_metrics = history["validation"]

    metric_titles = [
        ("dice", "Dice"),
        ("iou", "IoU"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("specificity", "Specificity"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.ravel()
    has_any_train_curve = train_metrics is not None and any(has_valid_series(train_metrics.get(key)) for key, _ in metric_titles)

    for ax, (key, title) in zip(axes, metric_titles):
        val_curve = val_metrics.get(key, [])
        train_curve = train_metrics.get(key, []) if train_metrics is not None else None

        if has_valid_series(train_curve):
            ax.plot(epochs, train_curve, marker="o", linewidth=1.8, markersize=5, label="Train", color="#1f77b4")

        ax.plot(epochs, val_curve, marker="s", linewidth=1.8, markersize=5, label="Validation", color="#d62728")
        ax.set_title(f"Train vs Validation {title}" if has_valid_series(train_curve) else f"Validation {title}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(
        "Training vs Validation Metrics Over Epochs" if has_any_train_curve else "Validation Metrics Over Epochs",
        fontsize=14,
        fontweight="bold"
    )
    fig.supxlabel("Epoch")
    fig.supylabel("Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"All-metrics validation curve saved to: {output_path}")
    plt.show()


def plot_key_metrics_with_loss(
    history_path="validation_history.json",
    output_path="validation_key_metrics_loss_curve.png"
):
    """Plot Dice, IoU, F1, and validation-time loss at each validation epoch."""
    history = load_validation_history(history_path)
    epochs = history["epochs"]
    train_metrics = history["train"]
    val_metrics = history["validation"]
    losses = history["loss"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()

    metric_specs = [
        ("dice", "Dice", "#d62728"),
        ("iou", "IoU", "#1f77b4"),
        ("f1", "F1", "#2ca02c"),
    ]

    has_any_train_curve = train_metrics is not None and any(has_valid_series(train_metrics.get(key)) for key, _, _ in metric_specs)

    for ax, (key, title, color) in zip(axes[:3], metric_specs):
        val_values = val_metrics.get(key, [])
        train_values = train_metrics.get(key, []) if train_metrics is not None else None

        if has_valid_series(train_values):
            ax.plot(epochs, train_values, marker="o", linewidth=2, markersize=6, color="#1f77b4", label="Train")

        ax.plot(epochs, val_values, marker="s", linewidth=2, markersize=6, color=color, label="Validation")
        ax.set_title(f"Train vs Validation {title}" if has_valid_series(train_values) else f"Validation {title}")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

        if has_valid_series(train_values):
            ax.legend(fontsize=9)

    loss_ax = axes[3]
    if losses is None:
        loss_ax.text(0.5, 0.5, "Loss unavailable", ha="center", va="center", transform=loss_ax.transAxes)
    else:
        loss_ax.plot(epochs, losses, marker="s", linewidth=2, markersize=6, color="#9467bd")
    loss_ax.set_title("Training Loss At Validation Epochs")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Score")
    axes[2].set_xlabel("Epoch")
    fig.suptitle(
        "Training vs Validation Dice / IoU / F1 + Loss" if has_any_train_curve else "Validation Dice / IoU / F1 + Loss",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Key metrics curve saved to: {output_path}")
    plt.show()


def load_training_loss(history_path="training_loss.json"):
    """Load training loss from training_loss.json only."""
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"{history_path} not found. Please run training to generate it."
        )

    with open(history_path, "r") as f:
        history = json.load(f)

    if not history:
        raise ValueError(f"{history_path} is empty")

    epochs = []
    losses = []
    for item in history:
        if isinstance(item, dict) and "epoch" in item:
            # support both "loss" (old) and "train_loss" (new) key names
            loss_val = item.get("train_loss", item.get("loss"))
            if loss_val is not None:
                epochs.append(int(item["epoch"]))
                losses.append(float(loss_val))

    if not epochs:
        raise ValueError(f"No epoch/loss records found in {history_path}")

    print(f"Loaded training loss from: {history_path}")
    return epochs, losses


def plot_training_loss(output_path="training_loss_curve.png"):
    """Plot training loss curve from training_loss.json."""
    epochs, losses = load_training_loss()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, losses, linewidth=2, color="#9467bd", label="Training Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Training loss curve saved to: {output_path}")
    plt.show()


def plot_combined_loss(
    history_path="validation_history.json",
    output_path="combined_loss_curve.png"
):
    """Plot training loss and validation loss on the same graph."""
    # Load training loss
    train_epochs, train_losses = load_training_loss()

    # Load validation loss (val_loss field)
    history = load_validation_history(history_path)
    val_epochs = history["epochs"]
    val_losses = history["val_loss"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot training loss (every epoch)
    ax.plot(
        train_epochs,
        train_losses,
        linewidth=2,
        color="#9467bd",
        label="Training Loss",
        alpha=0.8
    )

    # Plot validation loss (only at validation epochs)
    if val_losses and any(not np.isnan(float(l)) for l in val_losses if l is not None):
        clean_epochs = [e for e, l in zip(val_epochs, val_losses) if l is not None and not np.isnan(float(l))]
        clean_losses = [float(l) for l in val_losses if l is not None and not np.isnan(float(l))]
        ax.plot(
            clean_epochs,
            clean_losses,
            marker="o",
            linewidth=2,
            markersize=7,
            color="#d62728",
            label="Validation Loss",
            alpha=0.8
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Combined loss curve saved to: {output_path}")
    plt.show()


def compare_models(*report_files):
    """
    比较多个验证报告
    
    Usage:
        compare_models("report_v1.txt", "report_v2.txt", "report_v3.txt")
    """
    import re
    
    results = {}
    
    for report_file in report_files:
        if not os.path.exists(report_file):
            print(f"Warning: {report_file} not found")
            continue
        
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Extract metrics
        dice_match = re.search(r'Dice \(mean \± std\): ([\d.]+)', content)
        iou_match = re.search(r'IoU \(mean \± std\): ([\d.]+)', content)
        f1_match = re.search(r'F1-Score \(mean \± std\): ([\d.]+)', content)
        
        if dice_match:
            results[report_file] = {
                'Dice': float(dice_match.group(1)),
                'IoU': float(iou_match.group(1)) if iou_match else None,
                'F1': float(f1_match.group(1)) if f1_match else None
            }
    
    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<30} {'Dice':<15} {'IoU':<15} {'F1':<15}")
    print("-"*60)
    
    for model, metrics in results.items():
        print(f"{model:<30} {metrics['Dice']:<15.4f} {metrics['IoU']:<15.4f} {metrics['F1']:<15.4f}")
    
    # Show best
    best_model = max(results.items(), key=lambda x: x[1]['Dice'])
    print("-"*60)
    print(f"🏆 Best: {best_model[0]} (Dice: {best_model[1]['Dice']:.4f})")


def generate_summary_table():
    """
    从验证报告生成汇总表
    """
    if not os.path.exists("validation_report.txt"):
        print("validation_report.txt not found!")
        return
    
    import re
    
    with open("validation_report.txt", "r") as f:
        lines = f.readlines()
    
    # Find per-sample metrics section
    per_sample_start = None
    for i, line in enumerate(lines):
        if "Per-Sample Metrics:" in line:
            per_sample_start = i
            break
    
    if per_sample_start is None:
        print("Could not parse validation_report.txt")
        return
    
    # Extract per-sample data
    samples = []
    current_sample = {}
    
    for line in lines[per_sample_start:]:
        if line.strip().startswith("Global Metrics"):
            break
        
        # Check if new sample (has .tif in name)
        if ".tif" in line and not line.startswith(" "):
            if current_sample:
                samples.append(current_sample)
            current_sample = {'name': line.strip()}
        elif ":" in line and "dice" in line.lower():
            match = re.search(r'(\w+): ([\d.]+)', line)
            if match:
                current_sample[match.group(1)] = float(match.group(2))
    
    # Print table
    if samples:
        print("\n" + "="*80)
        print("PER-SAMPLE METRICS")
        print("="*80)
        print(f"{'Sample':<30} {'Dice':<12} {'IoU':<12} {'F1':<12} {'Recall':<12}")
        print("-"*80)
        
        for sample in samples:
            print(f"{sample.get('name', 'N/A'):<30} "
                  f"{sample.get('dice', 0):<12.4f} "
                  f"{sample.get('iou', 0):<12.4f} "
                  f"{sample.get('f1', 0):<12.4f} "
                  f"{sample.get('recall', 0):<12.4f}")


if __name__ == "__main__":
    import sys

    # Supported CLI options:
    #   allmetrics   -> Plot 6 validation metrics (Dice/IoU/F1/Precision/Recall/Specificity)
    #   loss         -> Plot training loss only (from training_loss.json)
    #   losscompare  -> Plot training loss vs validation loss on one figure
    #   table        -> Print summary table parsed from validation_report.txt
    # Notes:
    #   1) Make sure to run training and validation first to generate the necessary JSON and report files.
    #   2) Unknown options will print usage and exit.

    usage = (
        "Usage:\n"
        "  python plot_validation.py allmetrics\n"
        "  python plot_validation.py loss\n"
        "  python plot_validation.py losscompare\n"
        "  python plot_validation.py table"
    )

    if len(sys.argv) > 1:
        if sys.argv[1] == "table":
            generate_summary_table()
        elif sys.argv[1] == "loss":
            plot_training_loss()
        elif sys.argv[1] == "allmetrics":
            plot_all_metrics_history()
        elif sys.argv[1] == "losscompare":
            plot_combined_loss()
        else:
            print(usage)
    else:
        print(usage)

    # Run table generation only when explicitly requested via `table` option.
