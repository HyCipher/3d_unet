import json

import numpy as np
import torch
from validations.metrics import (  # pyright: ignore[reportMissingImports]
    dice_coefficient,
    iou_score,
    precision_recall_f1_specificity,
)


def sliding_window_inference_val(
    volume,
    model,
    patch_size=(8, 512, 512),
    stride=(2, 64, 64),
    device="cuda",
):
    """滑动窗口推理（用于验证）"""
    model.eval()
    z_dim, h_dim, w_dim = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = np.zeros((z_dim, h_dim, w_dim), dtype=np.float32)
    count_map = np.zeros((z_dim, h_dim, w_dim), dtype=np.float32)

    with torch.no_grad():
        for z in range(0, z_dim, sd):
            z0 = min(z, z_dim - pd)
            for y in range(0, h_dim - ph + 1, sh):
                y0 = min(y, h_dim - ph)
                for x in range(0, w_dim - pw + 1, sw):
                    x0 = min(x, w_dim - pw)

                    patch = volume[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
                    patch = (patch - patch.mean()) / (patch.std() + 1e-8)

                    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
                    patch = patch.to(device)

                    pred = model(patch)
                    pred = torch.sigmoid(pred)
                    pred = pred.cpu().numpy()[0, 0]

                    output[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += pred
                    count_map[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += 1

    output /= np.maximum(count_map, 1e-8)
    return output


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

    with torch.no_grad():
        for idx in range(total_volumes):
            vol = dataset.volumes[idx]
            lab = dataset.labels[idx]

            print(f"  validating volume {idx+1}/{total_volumes} ...")

            if criterion is not None:
                pd_s, ph_s, pw_s = patch_size
                sd_s, sh_s, sw_s = stride
                z_dim, h_dim, w_dim = vol.shape
                patch_losses = []
                for z in range(0, z_dim, sd_s):
                    z0 = min(z, z_dim - pd_s)
                    for y in range(0, h_dim - ph_s + 1, sh_s):
                        y0 = min(y, h_dim - ph_s)
                        for x in range(0, w_dim - pw_s + 1, sw_s):
                            x0 = min(x, w_dim - pw_s)
                            xp = vol[z0 : z0 + pd_s, y0 : y0 + ph_s, x0 : x0 + pw_s].copy()
                            yp = lab[z0 : z0 + pd_s, y0 : y0 + ph_s, x0 : x0 + pw_s].copy()
                            xp = (xp - xp.mean()) / (xp.std() + 1e-8)
                            yp = (yp > 0).astype(np.float32)
                            xt = torch.from_numpy(xp).unsqueeze(0).unsqueeze(0).float().to(device)
                            yt = torch.from_numpy(yp).unsqueeze(0).unsqueeze(0).float().to(device)
                            logits = model(xt)
                            patch_losses.append(criterion(logits, yt).item())
                loss_values.append(float(np.mean(patch_losses)))

            prob_map = sliding_window_inference_val(
                vol, model, patch_size=patch_size, stride=stride, device=device
            )

            pred_seg = (prob_map > threshold).astype(np.uint8)
            gt_seg = (lab > 0).astype(np.uint8)

            dice = dice_coefficient(pred_seg, gt_seg)
            dice_scores.append(dice)

            iou = iou_score(pred_seg, gt_seg)
            iou_scores.append(iou)

            precision, recall, f1, specificity = precision_recall_f1_specificity(
                pred_seg,
                gt_seg,
            )

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
