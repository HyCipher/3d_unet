def get_control_panel():
    """Centralized training/validation hyperparameters."""
    return {
        "validate_every": 10,
        "eval_train_set": True,
        "max_val_volumes": None,
        "val_patch_size": (16, 512, 512),
        "val_stride": (8, 256, 256),
        "val_threshold": 0.1,
        "dice_weight": 0.8,
        "focal_weight": 1.0,
        "num_epochs": 50,
        "loss_type": "bce",  # "bce" | "focal" | "dicefocal"
    }
