def get_validation_config():
    """Centralized defaults for standalone validation runs."""
    return {
        "model_path": "./model_results/run_20260530_192452/unet_3d_epoch_50.pth",
        "val_img_dir": "data/validation/images",
        "val_label_dir": "data/validation/labels",
        "patch_size": (8, 512, 512),
        "stride": (4, 256, 256),
        "threshold": 0.1,
        "dust_remove_min_size": 128,
        "eval_affinity": True,
        "affinity_offsets": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        "loss_type": "bce",  # "bce" | "focal" | "dicefocal"
        "save_results": True,   # Save pred/prob tif files to validation_results
        "wandb": False,          # Use Weights & Biases for logging
        "wandb_project": "c_elegans_3d_unet_validation",
        "wandb_run_name": f"run_20260530_192452_e50_tra",  # Customize as needed
    }