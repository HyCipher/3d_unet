import os
import torch
from models.detect import UNet


def init_model_and_lr(device, pretrained_path="model_results/run_20260526_213751-100134/unet_3d_best.pth"):
    """Create model and optionally load a pretrained checkpoint."""
    model = UNet().to(device)
    if os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path))
            print(f"Loaded pre-trained model from {pretrained_path}")
            return model, 1e-4, True
        except RuntimeError as e:
            print(
                "Checkpoint not compatible with current normalization setup; "
                f"training from scratch. Details: {e}"
            )

    print("No pre-trained model found, starting from scratch")
    return model, 1e-4, False
