import numpy as np
import torch


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
    avg_loss = float(np.mean(patch_losses)) if patch_losses else None
    return output, pred_seg, avg_loss