import tifffile as tiff
import numpy as np
import torch

from training.axis_utils import normalize_to_zyx, zyx_to_hwz
from models import SepUNet
from evaluation.inference import sliding_window_inference


# =========================
# Main inference function
# =========================
def infer_one_volume(
    img_path,
    model_path,
    save_path,
    patch_size=(8, 512, 512),
    stride=(2, 64, 64),
    threshold=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = SepUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # load tif (H,W,Z)
    vol = tiff.imread(img_path).astype(np.float32)

    vol, _ = normalize_to_zyx(vol, img_path, patch_size)

    print("Volume shape:", vol.shape)

    prob_map, seg, _ = sliding_window_inference(
        vol,
        None,
        model,
        patch_size=patch_size,
        stride=stride,
        threshold=threshold,
        device=device,
    )

    # Save as tif (back to H,W,Z)
    seg_save = zyx_to_hwz((seg * 255).astype(np.uint8), save_path)
    tiff.imwrite(save_path, seg_save)

    prob_map_save = zyx_to_hwz(prob_map, "prob_map.tif")
    tiff.imwrite("prob_map.tif", prob_map_save.astype(np.float32))

    print("Saved segmentation to:", save_path)
    return prob_map


# =========================
# Run
# =========================
if __name__ == "__main__":
    prob_map = infer_one_volume(
        img_path="/BiO/Live/rooter/Downloads/C_elegans_UNet/3d_unet/data/training/images",
        model_path="/BiO/Live/rooter/Downloads/C_elegans_UNet/3d_unet/model_results/run_20260515_130430/unet_3d_best.pth",
        save_path="result_seg.tif",
        patch_size=(16, 512, 512),
        stride=(4, 128, 128),
    )
