from config.aug_config import get_aug_config


def build_aug_wandb_config(prefix="aug/"):
    """Build a wandb-friendly config dict for augmentation hyperparameters."""
    aug_cfg = get_aug_config()
    return {
        f"{prefix}prob_geometric": aug_cfg["prob_geometric"],
        f"{prefix}prob_intensity": aug_cfg["prob_intensity"],
        f"{prefix}prob_artifact": aug_cfg["prob_artifact"],
        f"{prefix}translate_min_shift": str(aug_cfg["translate_min_shift"]),
        f"{prefix}translate_max_shift": str(aug_cfg["translate_max_shift"]),
        f"{prefix}blackpad_prob": aug_cfg["blackpad_prob"],
        f"{prefix}blackpad_pad_ratio_range": str(aug_cfg["blackpad_pad_ratio_range"]),
        f"{prefix}contrast_range": str(aug_cfg["contrast_range"]),
        f"{prefix}brightness_range": str(aug_cfg["brightness_range"]),
        f"{prefix}gamma_log2_range": str(aug_cfg["gamma_log2_range"]),
        f"{prefix}gaussian_noise_std": aug_cfg["gaussian_noise_std"],
        f"{prefix}section_intensity_shift_std": aug_cfg["section_intensity_shift_std"],
        f"{prefix}block_prob": aug_cfg["block_prob"],
        f"{prefix}block_shift": aug_cfg["block_shift"],
        f"{prefix}darkline_width_range": str(aug_cfg["darkline_width_range"]),
        f"{prefix}missing_section_max": aug_cfg["missing_section_max"],
        f"{prefix}elastic_alpha": aug_cfg["elastic_alpha"],
        f"{prefix}elastic_sigma": aug_cfg["elastic_sigma"],
    }
