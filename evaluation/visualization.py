import numpy as np
import matplotlib.pyplot as plt


def save_validation_visualization(
    volume,
    label,
    pred_seg,
    prob_map,
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
    axes[1, 1].set_title("Ground Truth Overlay")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img_slice, cmap="gray")
    axes[1, 2].imshow(pred_slice, cmap="Blues", alpha=0.4)
    axes[1, 2].set_title("Prediction Overlay")
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig
