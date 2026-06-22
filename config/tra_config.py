def tra_hyper():
    """Centralized training/validation hyperparameters."""
    return {
        "project": "c_elegans_3d_unet",
        "architecture": "3D-Sep-UNet",
        "train_img_dir": "data/training/images",
        "train_label_dir": "data/training/labels",
        "val_img_dir": "data/validation/images",
        "val_label_dir": "data/validation/labels",
        "patch_size": (8, 512, 512),
        "val_patches_per_volume": 50,
        "batch_size": 2,
        "num_workers": 4,
        "num_epochs": 50,
        "max_val_volumes": None,
        "val_patch_size": (8, 512, 512),
        "val_stride": (4, 256, 256),
        "val_threshold": 0.1,
        "dust_remove_min_size": 128,
        "loss_type": "bce",  # "bce" | "focal" | "dicefocal"
        "save_every": 10,
        "eval_train_set": False, # Set to True to evaluate training set each epoch (slows down training)
        "pos_weight_cap": 10.0, # Cap for pos_weight to prevent instability from extreme class imbalance
        "grad_clip_norm": 1.0,  # Gradient clipping to prevent exploding gradients
        "disable_aug_last_epochs": 8, # Number of final epochs to disable augmentation (to reduce late-stage noise)
        "dice_weight": 0.8,
        "focal_weight": 1.0,
        
        # --- Patch sampling ratios (must sum <= 1.0; remainder is random) ---
        "pos_sample_ratio": 0.2,            # Foreground (GT > 0) centered patches
        "edge_sample_ratio": 0.1,           # GT boundary voxel centered patches
        "hard_negative_enable": True,      # Enable hard-negative/positive sampling
        "hard_negative_dir": "data/hard_sampling/hard_negative_masks",          # Path to hard_negative_masks/ directory
        "hard_negative_sample_ratio": 0.2,  # Predicted CC with zero GT overlap
        "hard_positive_sample_ratio": 0.1,  # GT CC with zero prediction overlap
        "hardest_negative": False,          # Use eroded core of hard-negative mask
        "hardest_negative_erosion_iters": 1,
    }