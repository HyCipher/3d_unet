def tra_hyper():
    """Centralized training/validation hyperparameters."""
    return {
        "project": "c_elegans_3d_unet",
        "architecture": "3D UNet",
        "train_img_dir": "data/training/images",
        "train_label_dir": "data/training/labels",
        "val_img_dir": "data/validation/images",
        "val_label_dir": "data/validation/labels",
        "patch_size": (8, 512, 512),
        "patches_per_volume": 500,
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
        "validate_every": 10,
        "eval_train_set": False, # Set to True to evaluate training set each epoch (slows down training)
        "hard_negative_enable": True,
        "hard_negative_dir": "data/training_hn/hard_negative_masks",
        "hard_negative_sample_ratio": 0.5,
        "hardest_negative": False,
        "hardest_negative_erosion_iters": 1,
        "pos_weight_cap": 10.0, # Cap for pos_weight to prevent instability from extreme class imbalance
        "grad_clip_norm": 1.0,  # Gradient clipping to prevent exploding gradients
        "disable_aug_last_epochs": 8, # Number of final epochs to disable augmentation (to reduce late-stage noise)
        "dice_weight": 0.8,
        "focal_weight": 1.0,
    }