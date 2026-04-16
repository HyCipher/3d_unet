"""
阈值优化脚本：根据验证集找到最优的分类阈值
用来最大化F1分数或其他指标
"""
import numpy as np
import torch
import glob
import os
import tifffile as tiff
from detect import UNet
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

def sliding_window_inference(volume, model, patch_size=(8, 512, 512), stride=(2, 64, 64), device="cuda"):
    """Sliding window inference on full volume."""
    model.eval()
    Z, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = np.zeros((Z, H, W), dtype=np.float32)
    count_map = np.zeros((Z, H, W), dtype=np.float32)

    with torch.no_grad():
        for z in range(0, Z, sd):
            z0 = min(z, Z - pd)
            for y in range(0, H - ph + 1, sh):
                y0 = min(y, H - ph)
                for x in range(0, W - pw + 1, sw):
                    x0 = min(x, W - pw)

                    patch = volume[z0:z0+pd, y0:y0+ph, x0:x0+pw]
                    patch = (patch - patch.mean()) / (patch.std() + 1e-8)

                    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    pred = torch.sigmoid(model(patch))
                    pred = pred.cpu().numpy()[0, 0]

                    output[z0:z0+pd, y0:y0+ph, x0:x0+pw] += pred
                    count_map[z0:z0+pd, y0:y0+ph, x0:x0+pw] += 1

    output /= np.maximum(count_map, 1e-8)
    return output


def find_optimal_threshold(val_img_dir, val_label_dir, model_path, device="cuda"):
    """Find optimal threshold by maximizing F1 score."""
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    img_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.tif")))
    label_paths = sorted(glob.glob(os.path.join(val_label_dir, "*.tif")))

    all_probs = []
    all_labels = []

    print(f"Processing {len(img_paths)} validation volumes...")
    for i, (img_path, label_path) in enumerate(zip(img_paths, label_paths)):
        print(f"  Volume {i+1}/{len(img_paths)}: {os.path.basename(img_path)}")

        vol = tiff.imread(img_path).astype(np.float32)
        lab = tiff.imread(label_path).astype(np.float32)

        # (H,W,Z) -> (Z,H,W)
        vol = np.transpose(vol, (2, 0, 1))
        lab = np.transpose(lab, (2, 0, 1))

        # Get predictions
        prob_map = sliding_window_inference(vol, model, device=device)

        all_probs.append(prob_map.flatten())
        all_labels.append((lab > 0).astype(np.float32).flatten())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    print(f"\nPositive samples: {all_labels.sum()}/{len(all_labels)} ({100*all_labels.mean():.2f}%)")

    # Calculate PR curve
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    print(f"\n📊 最优阈值: {best_threshold:.4f}")
    print(f"   F1-Score: {best_f1:.4f}")
    print(f"   Precision: {best_precision:.4f}")
    print(f"   Recall: {best_recall:.4f}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PR Curve
    axes[0].plot(recall, precision, 'b-', label='PR Curve')
    axes[0].scatter([best_recall], [best_precision], color='red', s=100, label=f'Best (threshold={best_threshold:.3f})')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve')
    axes[0].legend()
    axes[0].grid(True)

    # F1 vs Threshold
    axes[1].plot(pr_thresholds, f1_scores[:len(pr_thresholds)], 'g-', linewidth=2)
    axes[1].scatter([best_threshold], [best_f1], color='red', s=100, label=f'Best threshold={best_threshold:.3f}')
    axes[1].set_xlabel('Classification Threshold')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('F1-Score vs Threshold')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=150, bbox_inches='tight')
    print("\n✅ 已保存图表: threshold_optimization.png")

    return best_threshold, best_f1, best_precision, best_recall


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_threshold, best_f1, best_precision, best_recall = find_optimal_threshold(
        val_img_dir="data/validation/images",
        val_label_dir="data/validation/labels",
        model_path="./models/unet_3d_best.pth",
        device=device
    )

    print(f"\n💡 建议：在config.py中将 val_threshold 改为 {best_threshold:.4f}")
