import datetime


def get_validation_config():
    """Centralized defaults for standalone validation runs."""
    return {
        "model_path": "./models/run_20260422_001521/unet_3d_best.pth",
        "val_img_dir": "data/validation/images",
        "val_label_dir": "data/validation/labels",
        "patch_size": (8, 512, 512),
        "stride": (8, 256, 256),
        "threshold": 0.1,
        "loss_type": "bce",  # "bce" | "focal" | "dicefocal"
        "save_results": True,   # Save pred/prob tif files to validation_results
        "plot_curves": True,    # Plot PR/ROC curves
        "visualize": True,      # Visualize predictions
        "wandb": True,          # Use Weights & Biases for logging
        "wandb_project": "c_elegans_3d_unet_validation",
        "wandb_run_name": f"run_20260422_001521",
    }
