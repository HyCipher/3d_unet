import json

import numpy as np
import torch
from evaluation.inference import sliding_window_inference  # pyright: ignore[reportMissingImports]
from validate.metrics import (  # pyright: ignore[reportMissingImports]
    dice_coefficient,
    iou_score,
    precision_recall_f1_specificity,
)


def validate_with_full_metrics(
    model,
    dataset,
    device,
    patch_size=(8, 512, 512),
    stride=(2, 64, 64),
    threshold=0.5,
    criterion=None,
):
    """计算完整指标：Dice, IoU, F1, Precision, Recall, Specificity（可选 val loss）"""
    model.eval()
    dice_scores = []
    iou_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []
    loss_values = []

    total_volumes = len(dataset.volumes)
    print(f"Validation start: {total_volumes} volumes, patch={patch_size}, stride={stride}")

    for idx in range(total_volumes):
        vol = dataset.volumes[idx]
        lab = dataset.labels[idx]

        prob_map, pred_seg, avg_loss = sliding_window_inference(
            vol,
            lab,
            model,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            device=device,
            criterion=criterion,
        )

        if avg_loss is not None:
            loss_values.append(avg_loss)

        gt_seg = (lab > 0).astype(np.uint8)

        dice_scores.append(dice_coefficient(pred_seg, gt_seg))
        iou_scores.append(iou_score(pred_seg, gt_seg))

        precision, recall, f1, specificity = precision_recall_f1_specificity(pred_seg, gt_seg)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        specificity_scores.append(specificity)

    result = {
        "dice": np.mean(dice_scores),
        "iou": np.mean(iou_scores),
        "f1": np.mean(f1_scores),
        "precision": np.mean(precision_scores),
        "recall": np.mean(recall_scores),
        "specificity": np.mean(specificity_scores),
    }
    if loss_values:
        result["loss"] = float(np.mean(loss_values))
    return result


def save_validation_history(history, history_path="validation_history.json"):
    """Persist validation history after each evaluation so curves can be plotted later."""
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
