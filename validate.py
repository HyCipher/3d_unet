import argparse
import glob
import os

import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from nets.detect import UNet
from losses import FocalLoss, DiceFocalLoss
from validate.metrics import dice_coefficient, iou_score, precision_recall_f1_specificity
from tracking import (
    log_pr_roc_to_wandb,
    log_sample_to_wandb,
    log_generated_files_to_wandb,
    log_summary_table_to_wandb,
)


def _gen_starts(length, patch, stride):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def sliding_window_inference(
    volume,
    label,
    model,
    patch_size=(16, 512, 512),
    stride=(8, 256, 256),
    threshold=0.5,
    device="cuda",
    criterion=None,
):
    model.eval()
    z_len, h_len, w_len = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = np.zeros((z_len, h_len, w_len), dtype=np.float32)
    count_map = np.zeros((z_len, h_len, w_len), dtype=np.float32)
    patch_losses = []

    z_starts = _gen_starts(z_len, pd, sd)
    y_starts = _gen_starts(h_len, ph, sh)
    x_starts = _gen_starts(w_len, pw, sw)

    with torch.no_grad():
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    patch = volume[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw].copy()
                    patch = (patch - patch.mean()) / (patch.std() + 1e-8)

                    xt = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    logits = model(xt)
                    probs = torch.sigmoid(logits)

                    output[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += probs.cpu().numpy()[0, 0]
                    count_map[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += 1.0

                    if criterion is not None and label is not None:
                        y_patch = label[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw].copy()
                        y_patch = (y_patch > 0).astype(np.float32)
                        yt = torch.from_numpy(y_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                        patch_losses.append(criterion(logits, yt).item())

    output /= np.maximum(count_map, 1e-8)
    pred_seg = (output > threshold).astype(np.uint8)
    avg_loss = float(np.mean(patch_losses)) if patch_losses else None
    return output, pred_seg, avg_loss


def load_validation_pairs(img_dir, label_dir):
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.tif")))

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No .tif files found in {img_dir}")
    if len(img_paths) != len(label_paths):
        raise ValueError(
            f"Image/label count mismatch: {len(img_paths)} images vs {len(label_paths)} labels"
        )

    return list(zip(img_paths, label_paths))


def save_prediction_results(prob_map, pred_seg, img_path, out_dir="validation_results"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(img_path)

    # Save as (H, W, Z) for consistency with source tif layout.
    pred_hwz = np.transpose(pred_seg.astype(np.uint8) * 255, (1, 2, 0))
    prob_hwz = np.transpose(prob_map.astype(np.float32), (1, 2, 0))

    pred_path = os.path.join(out_dir, f"pred_{base}")
    prob_path = os.path.join(out_dir, f"prob_{base}")
    tiff.imwrite(pred_path, pred_hwz)
    tiff.imwrite(prob_path, prob_hwz)


def sample_for_curves(gt_seg, prob_map, max_points=300000):
    y_true = gt_seg.reshape(-1).astype(np.uint8)
    y_score = prob_map.reshape(-1).astype(np.float32)

    if y_true.size > max_points:
        step = int(np.ceil(y_true.size / max_points))
        y_true = y_true[::step]
        y_score = y_score[::step]

    return y_true, y_score


def save_validation_visualization(
    volume,
    label,
    pred_seg,
    prob_map,
    out_path="validation_visualization.png",
):
    # Use center slice for a quick visual sanity check.
    z_mid = volume.shape[0] // 2

    img_slice = volume[z_mid]
    gt_slice = (label[z_mid] > 0).astype(np.uint8)
    pred_slice = pred_seg[z_mid].astype(np.uint8)
    prob_slice = prob_map[z_mid]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(img_slice, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_slice, cmap="gray")
    axes[0, 1].set_title("Label")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_slice, cmap="gray")
    axes[0, 2].set_title("Prediction")
    axes[0, 2].axis("off")

    im = axes[1, 0].imshow(prob_slice, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Probability")
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(img_slice, cmap="gray")
    axes[1, 1].imshow(gt_slice, cmap="Reds", alpha=0.4)
    axes[1, 1].set_title("Label Overlay")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img_slice, cmap="gray")
    axes[1, 2].imshow(pred_slice, cmap="Blues", alpha=0.4)
    axes[1, 2].set_title("Prediction Overlay")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Visualization saved to: {out_path}")
    return out_path


def evaluate_model(
    model_path,
    val_img_dir,
    val_label_dir,
    patch_size=(16, 512, 512),
    stride=(8, 256, 256),
    threshold=0.5,
    loss_type="bce",
    save_results=True,
    plot_curves=True,
    save_visualization=True,
    wandb_run=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = None
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif loss_type == "dicefocal":
        criterion = DiceFocalLoss(alpha=0.25, gamma=2.0, dice_weight=0.8, focal_weight=1.5)

    pairs = load_validation_pairs(val_img_dir, val_label_dir)

    dice_list = []
    iou_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    loss_list = []
    curve_true = []
    curve_score = []
    visualization_saved = False
    visualization_path = None

    print(f"Model: {model_path}")
    print(f"Validation samples: {len(pairs)}")
    print(f"Patch size: {patch_size}, stride: {stride}, threshold: {threshold}")

    for i, (img_path, label_path) in enumerate(pairs, start=1):
        sample_name = os.path.basename(img_path)
        vol = tiff.imread(img_path).astype(np.float32)
        lab = tiff.imread(label_path).astype(np.float32)

        # (H, W, Z) -> (Z, H, W)
        vol = np.transpose(vol, (2, 0, 1))
        lab = np.transpose(lab, (2, 0, 1))

        prob_map, pred_seg, sample_loss = sliding_window_inference(
            vol,
            lab,
            model,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            device=device,
            criterion=criterion,
        )

        gt_seg = (lab > 0).astype(np.uint8)

        dice = dice_coefficient(pred_seg, gt_seg)
        iou = iou_score(pred_seg, gt_seg)

        precision, recall, f1, specificity = precision_recall_f1_specificity(pred_seg, gt_seg)

        dice_list.append(dice)
        iou_list.append(iou)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)

        if sample_loss is not None:
            loss_list.append(sample_loss)

        if save_results:
            save_prediction_results(prob_map, pred_seg, img_path, out_dir="validation_results")

        if plot_curves:
            y_true, y_score = sample_for_curves(gt_seg, prob_map)
            curve_true.append(y_true)
            curve_score.append(y_score)

        if save_visualization and not visualization_saved:
            visualization_path = save_validation_visualization(
                volume=vol,
                label=lab,
                pred_seg=pred_seg,
                prob_map=prob_map,
                out_path="validation_visualization.png",
            )
            visualization_saved = True

        sample_metrics = {
            "dice": float(dice),
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "loss": float(sample_loss) if sample_loss is not None else None,
        }
        log_sample_to_wandb(
            wandb_run,
            sample_name,
            vol,
            lab,
            pred_seg,
            prob_map,
            sample_metrics,
            step=i,
        )

        print(
            f"[{i}/{len(pairs)}] {sample_name} | "
            f"Dice={dice:.4f}, IoU={iou:.4f}, F1={f1:.4f}, "
            f"P={precision:.4f}, R={recall:.4f}, Spec={specificity:.4f}"
            + (f", Loss={sample_loss:.4f}" if sample_loss is not None else "")
        )

        # Keep optional reference to probability map to ensure no linter warning for unused var in some setups.
        _ = prob_map

    if plot_curves and curve_true:
        y_true_all = np.concatenate(curve_true)
        y_score_all = np.concatenate(curve_score)
        log_pr_roc_to_wandb(wandb_run, y_true_all, y_score_all)

    summary = {
        "dice": float(np.mean(dice_list)),
        "iou": float(np.mean(iou_list)),
        "f1": float(np.mean(f1_list)),
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "specificity": float(np.mean(specificity_list)),
    }
    if loss_list:
        summary["loss"] = float(np.mean(loss_list))

    log_generated_files_to_wandb(
        wandb_run,
        visualization_path=visualization_path,
    )

    if wandb_run is not None:
        summary_payload = {f"validation/{key}": value for key, value in summary.items()}
        wandb_run.log(summary_payload)
        log_summary_table_to_wandb(wandb_run, summary)

    return summary


def save_report(model_path, summary):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    report_path = f"validation_report_{model_name}.txt"

    lines = [
        "=== Validation Summary ===",
        f"Model: {model_path}",
        f"Dice: {summary['dice']:.6f}",
        f"IoU: {summary['iou']:.6f}",
        f"F1: {summary['f1']:.6f}",
        f"Precision: {summary['precision']:.6f}",
        f"Recall: {summary['recall']:.6f}",
        f"Specificity: {summary['specificity']:.6f}",
    ]
    if "loss" in summary:
        lines.append(f"Validation Loss (mean): {summary['loss']:.6f}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report saved to: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one 3D UNet model on validation dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/run_20260420_150612/unet_3d_best.pth",
        help="Path to model",
    )
    parser.add_argument("--val-img-dir", type=str, default="data/validation/images")
    parser.add_argument("--val-label-dir", type=str, default="data/validation/labels")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(8, 512, 512))
    parser.add_argument("--stride", type=int, nargs=3, default=(8, 256, 256))
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["bce", "focal", "dicefocal"],
        default="bce",
        help="Loss used only for validation-loss reporting.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save pred/prob tif files to validation_results/",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_false",
        dest="save_results",
        help="Do not save pred/prob tif files.",
    )
    parser.add_argument(
        "--plot-curves",
        action="store_true",
        default=True,
        help="Generate PR/ROC curves PNG.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Save 6-panel visualization PNG.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Do not save visualization PNG.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Log validation metrics and slice images to wandb.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="c_elegans_3d_unet_validation",
        help="wandb project name used when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional wandb run name for this validation job.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    loss_type = None if args.loss_type == "none" else args.loss_type

    wandb_run = None
    if args.wandb:
        run_name = args.wandb_run_name
        if run_name is None:
            model_name = os.path.splitext(os.path.basename(args.model))[0]
            run_name = f"validate_{model_name}"

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_path": args.model,
                "val_img_dir": args.val_img_dir,
                "val_label_dir": args.val_label_dir,
                "patch_size": tuple(args.patch_size),
                "stride": tuple(args.stride),
                "threshold": args.threshold,
                "loss_type": loss_type,
                "save_results": args.save_results,
                "plot_curves": args.plot_curves,
                "save_visualization": args.visualize,
            },
            job_type="validation",
        )

    try:
        summary = evaluate_model(
            model_path=args.model,
            val_img_dir=args.val_img_dir,
            val_label_dir=args.val_label_dir,
            patch_size=tuple(args.patch_size),
            stride=tuple(args.stride),
            threshold=args.threshold,
            loss_type=loss_type,
            save_results=args.save_results,
            plot_curves=args.plot_curves,
            save_visualization=args.visualize,
            wandb_run=wandb_run,
        )

        print("\n=== Mean Metrics ===")
        print(f"Dice: {summary['dice']:.4f}")
        print(f"IoU: {summary['iou']:.4f}")
        print(f"F1: {summary['f1']:.4f}")
        print(f"Precision: {summary['precision']:.4f}")
        print(f"Recall: {summary['recall']:.4f}")
        print(f"Specificity: {summary['specificity']:.4f}")
        if "loss" in summary:
            print(f"Validation Loss: {summary['loss']:.4f}")

        save_report(args.model, summary)
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    # Default values (same meaning as VALIDATION_GUIDE.md):
    # model_path = "./models/unet_3d_best.pth"
    # img_dir = "data/validation/images"
    # label_dir = "data/validation/labels"
    # patch_size = (16, 512, 512)
    # stride = (8, 256, 256)
    # threshold = 0.5
    # save_results = True
    # plot_curves = False
    main()
