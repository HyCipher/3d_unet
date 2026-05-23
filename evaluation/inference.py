import numpy as np
import torch
from scipy import ndimage

from evaluation.postprocessing import remove_small_connected_components


def _log_cc_stats(pred_seg_before, pred_seg_after, threshold, dust_remove_min_size):
    """Print connected-component statistics before and after dust removal."""
    structure = ndimage.generate_binary_structure(3, 1)

    labeled_before, n_before = ndimage.label(pred_seg_before, structure=structure)
    sizes_before = np.bincount(labeled_before.ravel())[1:] if n_before > 0 else np.array([0])

    labeled_after, n_after = ndimage.label(pred_seg_after, structure=structure)
    sizes_after = np.bincount(labeled_after.ravel())[1:] if n_after > 0 else np.array([0])

    fg_before = int(pred_seg_before.sum())
    fg_after  = int(pred_seg_after.sum())

    print(
        f"[CC stats] threshold={threshold}  dust_min_size={dust_remove_min_size}\n"
        f"  Before dust remove : {n_before:5d} components | "
        f"fg voxels={fg_before:8d} | "
        f"max={sizes_before.max() if n_before else 0:8d} | "
        f"min={sizes_before.min() if n_before else 0:6d} | "
        f"median={int(np.median(sizes_before)) if n_before else 0:8d}\n"
        f"  After  dust remove : {n_after:5d} components | "
        f"fg voxels={fg_after:8d} | "
        f"max={sizes_after.max() if n_after else 0:8d} | "
        f"min={sizes_after.min() if n_after else 0:6d} | "
        f"median={int(np.median(sizes_after)) if n_after else 0:8d}"
    )


def gen_starts(length, patch, stride):
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
    dust_remove_min_size=0,
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

    z_starts = gen_starts(z_len, pd, sd)
    y_starts = gen_starts(h_len, ph, sh)
    x_starts = gen_starts(w_len, pw, sw)

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
    pred_seg_after = remove_small_connected_components(pred_seg, min_size=dust_remove_min_size)
    _log_cc_stats(pred_seg, pred_seg_after, threshold, dust_remove_min_size)
    avg_loss = float(np.mean(patch_losses)) if patch_losses else None
    return output, pred_seg_after, avg_loss