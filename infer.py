import tifffile as tiff
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nets.detect import UNet   # 你的 3D UNet

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import precision_recall_curve, average_precision_score
# =========================
# Sliding window inference
# =========================
def sliding_window_inference(
    volume,
    model,
    patch_size=(8, 512, 512),
    stride=(2, 64, 64),
    device="cuda"
):
    """
    volume: numpy array (Z, H, W)
    return: numpy array (Z, H, W) probability map
    """
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

                    # normalize（和训练一致）
                    # patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
                    patch = (patch - patch.mean()) / (patch.std() + 1e-8)

                    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
                    patch = patch.to(device)

                    pred = model(patch)
                    pred = torch.sigmoid(pred)   # [1,1,D,H,W]
                    pred = pred.cpu().numpy()[0, 0]

                    output[z0:z0+pd, y0:y0+ph, x0:x0+pw] += pred
                    count_map[z0:z0+pd, y0:y0+ph, x0:x0+pw] += 1

    output /= np.maximum(count_map, 1e-8)
    return output


# =========================
# Main inference function
# =========================
def infer_one_volume(
    img_path,
    model_path,
    save_path,
    patch_size=(8, 512, 512),
    stride=(2, 64, 64)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # load tif (H,W,Z)
    vol = tiff.imread(img_path).astype(np.float32)

    # (H,W,Z) → (Z,H,W)
    vol = np.transpose(vol, (2, 0, 1))

    print("Volume shape:", vol.shape)

    prob_map = sliding_window_inference(
        vol,
        model,
        patch_size=patch_size,
        stride=stride,
        device=device
    )

    # 二值化（可调阈值）
    seg = (prob_map > 0.1).astype(np.uint8) * 255

    # 保存为 tif（转回 H,W,Z）
    seg = np.transpose(seg, (1, 2, 0))
    tiff.imwrite(save_path, seg)
    prob_map_save = np.transpose(prob_map, (1, 2, 0))
    tiff.imwrite("prob_map.tif", prob_map_save.astype(np.float32))

    print("Saved segmentation to:", save_path)
    return prob_map


# =========================
# Run
# =========================
if __name__ == "__main__":
    prob_map = infer_one_volume(
        img_path="/BiO/Live/rooter/Downloads/C_elegans_UNet/3d_unet/data/validation/images/dauer_img_160_180.tif",
        model_path="/BiO/Live/rooter/Downloads/C_elegans_UNet/3d_unet/models/unet_3d_epoch_50.pth",
        save_path="result_seg.tif",
        patch_size=(16, 512, 512),
        stride=(4, 128, 128)
    )