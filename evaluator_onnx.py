import os
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import tifffile as tiff
import torch
import wandb

from config.val_config import get_validation_config
from validate.metrics import dice_coefficient, iou_score, precision_recall_f1_specificity
from evaluation import (
    load_validation_pairs,
    save_prediction_results,
    sample_for_curves,
    save_validation_visualization,
)
from evaluation.inference import gen_starts
from evaluation.postprocessing import remove_small_connected_components
from evaluation.loss_factory import build_validation_criterion
from utils import (
    log_pr_roc_to_wandb,
    log_sample_table_to_wandb,
    log_summary_table_to_wandb,
)


VAL_CONFIG = get_validation_config()

def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_onnx_session(model_path):
    providers = ["CPUExecutionProvider"]
    if ort.get_device().upper() == "GPU":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def sliding_window_inference_onnx(
    volume,
    label,
    session,
    input_name,
    output_name,
    patch_size=(16, 512, 512),
    stride=(8, 256, 256),
    threshold=0.5,
    dust_remove_min_size=0,
    criterion=None,
):
    z_len, h_len, w_len = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = np.zeros((z_len, h_len, w_len), dtype=np.float32)
    count_map = np.zeros((z_len, h_len, w_len), dtype=np.float32)
    patch_losses = []

    z_starts = gen_starts(z_len, pd, sd)
    y_starts = gen_starts(h_len, ph, sh)
    x_starts = gen_starts(w_len, pw, sw)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = volume[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw].copy()
                patch = (patch - patch.mean()) / (patch.std() + 1e-8)

                x_np = patch[np.newaxis, np.newaxis, ...].astype(np.float32)
                logits_np = session.run([output_name], {input_name: x_np})[0]
                probs_np = _sigmoid_np(logits_np)

                output[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += probs_np[0, 0]
                count_map[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += 1.0

                if criterion is not None and label is not None:
                    y_patch = label[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw].copy()
                    y_patch = (y_patch > 0).astype(np.float32)
                    logits_t = torch.from_numpy(logits_np)
                    yt = torch.from_numpy(y_patch).unsqueeze(0).unsqueeze(0).float()
                    patch_losses.append(criterion(logits_t, yt).item())

    output /= np.maximum(count_map, 1e-8)
    pred_seg = (output > threshold).astype(np.uint8)
    pred_seg = remove_small_connected_components(pred_seg, min_size=dust_remove_min_size)
    avg_loss = float(np.mean(patch_losses)) if patch_losses else None
    return output, pred_seg, avg_loss

def evaluate_model_onnx(
    model_path,
    val_img_dir,
    val_label_dir,
    patch_size=VAL_CONFIG["patch_size"],
    stride=VAL_CONFIG["stride"],
    threshold=VAL_CONFIG["threshold"],
    dust_remove_min_size=VAL_CONFIG["dust_remove_min_size"],
    loss_type=VAL_CONFIG["loss_type"],
    save_results=VAL_CONFIG["save_results"],
    wandb_run=None,
):
    session, input_name, output_name = build_onnx_session(model_path)

    criterion = build_validation_criterion(loss_type)

    pairs = load_validation_pairs(val_img_dir, val_label_dir)

    dice_list = []
    iou_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    accuracy_list = []
    loss_list = []
    sample_rows = []
    curve_true = []
    curve_score = []

    for i, (img_path, label_path) in enumerate(pairs, start=1):
        sample_name = os.path.basename(img_path)
        vol = tiff.imread(img_path).astype(np.float32)
        lab = tiff.imread(label_path).astype(np.float32)

        vol = np.transpose(vol, (2, 0, 1))
        lab = np.transpose(lab, (2, 0, 1))

        prob_map, pred_seg, sample_loss = sliding_window_inference_onnx(
            vol,
            lab,
            session,
            input_name,
            output_name,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            dust_remove_min_size=dust_remove_min_size,
            criterion=criterion,
        )

        gt_seg = (lab > 0).astype(np.uint8)

        dice = dice_coefficient(pred_seg, gt_seg)
        iou = iou_score(pred_seg, gt_seg)
        precision, recall, f1, specificity, accuracy = precision_recall_f1_specificity(pred_seg, gt_seg)

        dice_list.append(dice)
        iou_list.append(iou)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        accuracy_list.append(accuracy)

        if sample_loss is not None:
            loss_list.append(sample_loss)

        if save_results:
            save_prediction_results(prob_map, pred_seg, img_path, out_dir="validation_results_onnx")

        y_true, y_score = sample_for_curves(gt_seg, prob_map)
        curve_true.append(y_true)
        curve_score.append(y_score)

        sample_image = None
        if wandb_run is not None:
            fig = save_validation_visualization(
                volume=vol,
                label=lab,
                pred_seg=pred_seg,
                prob_map=prob_map,
            )
            sample_image = wandb.Image(fig, caption=f"{sample_name} | val visualization")
            plt.close(fig)

        sample_rows.append(
            {
                "sample_index": i,
                "sample_name": sample_name,
                "dice": float(dice),
                "iou": float(iou),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "accuracy": float(accuracy),
                "loss": float(sample_loss) if sample_loss is not None else None,
                "sample_image": sample_image,
            }
        )

    if curve_true and wandb_run is not None:
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
        "accuracy": float(np.mean(accuracy_list)),
    }
    if loss_list:
        summary["loss"] = float(np.mean(loss_list))

    if wandb_run is not None:
        log_sample_table_to_wandb(wandb_run, sample_rows)
        log_summary_table_to_wandb(wandb_run, summary)

    return summary


def main():
    config = VAL_CONFIG
    model_path = config["model_path"]
    val_img_dir = config["val_img_dir"]
    val_label_dir = config["val_label_dir"]
    patch_size = tuple(config["patch_size"])
    stride = tuple(config["stride"])
    threshold = config["threshold"]
    dust_remove_min_size = config["dust_remove_min_size"]
    loss_type = config["loss_type"]
    save_results = config["save_results"]

    use_wandb = config["wandb"]
    wandb_project = config["wandb_project"]
    wandb_run_name = config["wandb_run_name"]

    if not model_path.endswith(".onnx"):
        raise ValueError(f"Expected .onnx model path, got: {model_path}")

    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_path": model_path,
                "val_img_dir": val_img_dir,
                "val_label_dir": val_label_dir,
                "patch_size": patch_size,
                "stride": stride,
                "threshold": threshold,
                "dust_remove_min_size": dust_remove_min_size,
                "loss_type": loss_type,
                "save_results": save_results,
            },
            job_type="validation-onnx",
            settings=wandb.Settings(silent=True, console="off"),
        )

    try:
        summary = evaluate_model_onnx(
            model_path=model_path,
            val_img_dir=val_img_dir,
            val_label_dir=val_label_dir,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            dust_remove_min_size=dust_remove_min_size,
            loss_type=loss_type,
            save_results=save_results,
            wandb_run=wandb_run,
        )

        print("\n=== Mean Metrics (ONNX) ===")
        print(f"Dice: {summary['dice']:.4f}")
        print(f"IoU: {summary['iou']:.4f}")
        print(f"F1: {summary['f1']:.4f}")
        print(f"Precision: {summary['precision']:.4f}")
        print(f"Recall: {summary['recall']:.4f}")
        print(f"Specificity: {summary['specificity']:.4f}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        if "loss" in summary:
            print(f"Validation Loss: {summary['loss']:.4f}")

    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
