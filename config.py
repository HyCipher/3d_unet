def get_control_panel():
    """Centralized training/validation hyperparameters."""
    return {
        "validate_every": 5,
        "eval_train_set": True,
        "max_val_volumes": None,
        "val_patch_size": (8, 512, 512),
        "val_stride": (8, 256, 256),
        "val_threshold": 0.35,
        "dice_weight": 0.8,
        "focal_weight": 1.0,
        "num_epochs": 50,
        "pos_weight_cap": 10.0,
        "grad_clip_norm": 1.0,
        "disable_aug_last_epochs": 8,
        "loss_type": "bce",  # "bce" | "focal" | "dicefocal"
    }
