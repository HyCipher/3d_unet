from config.aug_config import get_aug_config


def build_aug_wandb_config(prefix="aug/"):
    """Build a wandb-friendly config dict for augmentation hyperparameters."""
    aug_cfg = get_aug_config()
    return {
        f"{prefix}prob_geometric": aug_cfg["prob_geometric"],
        f"{prefix}prob_intensity": aug_cfg["prob_intensity"],
        f"{prefix}prob_artifact": aug_cfg["prob_artifact"],
        f"{prefix}prob_blackpad": aug_cfg["prob_blackpad"],
        f"{prefix}prob_flip": aug_cfg["prob_flip"],
        f"{prefix}prob_rotation_90": aug_cfg["prob_rotation_90"],
        f"{prefix}prob_translate": aug_cfg["prob_translate"],
        f"{prefix}prob_contrast": aug_cfg["prob_contrast"],
        f"{prefix}prob_gaussian_noise": aug_cfg["prob_gaussian_noise"],
        f"{prefix}prob_section_intensity_shift": aug_cfg["prob_section_intensity_shift"],
        f"{prefix}prob_block": aug_cfg["prob_block"],
        f"{prefix}prob_darkline": aug_cfg["prob_darkline"],
        f"{prefix}prob_missing_section": aug_cfg["prob_missing_section"],
        f"{prefix}prob_elastic": aug_cfg["prob_elastic"],
    }
