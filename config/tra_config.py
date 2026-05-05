def get_control_panel():
    """Centralized training/validation hyperparameters."""
    return {
        "validate_every": 10,
        "eval_train_set": True,
        "max_val_volumes": None,
        "val_patch_size": (8, 512, 512),
        "val_stride": (4, 256, 256),
        "val_threshold": 0.1,
        "dice_weight": 0.8,
        "focal_weight": 1.0,
        "num_epochs": 50,
        "pos_weight_cap": 10.0, # Cap for pos_weight to prevent instability from extreme class imbalance
        "grad_clip_norm": 1.0,  # Gradient clipping to prevent exploding gradients
        "disable_aug_last_epochs": 8,
        "loss_type": "bce",  # "bce" | "focal" | "dicefocal"
    }
