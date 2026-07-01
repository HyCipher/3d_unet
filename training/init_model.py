import os
import torch
from models import UNet3D, SepUNet, KiUNet, UNet, ResSepUNet


def init_model_and_lr(device, pretrained_path="model_results/unet_3d_best.pth"):
    """Create model and optionally load a pretrained checkpoint."""
    model = ResSepUNet().to(device)
    load_error = None

    if os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            print(f"Loaded pre-trained model from {pretrained_path}")
            return model, 1e-4, True
        except RuntimeError as e:
            load_error = e

    if load_error is not None:
        print(
            f"Found checkpoint at {pretrained_path} but failed to load; "
            f"training from scratch. Details: {load_error}"
        )
    else:
        print(f"No pre-trained model found at {pretrained_path}; training from scratch")

    return model, 1e-4, False