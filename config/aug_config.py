def get_aug_config():
    """Hyperparameters for data augmentation.
    prob_*      : probability that the entire group is applied.
    Ops inside each group are chosen uniformly at random when the group fires.
    """
    return {
        # ── group-level probabilities ──────────────────────────────────────
        "prob_geometric": 0.8,
        "prob_intensity": 0.8,
        "prob_artifact": 0.2,

        # ── geometric ops ─────────────────────────────────────────────────
        "translate_min_shift": (0, 1, 1),
        "translate_max_shift": (0, 3, 3),
        "blackpad_pad_ratio_range": (0.4, 0.9),
        "prob_blackpad": 0.15,
        "prob_flip": 1.0,
        "prob_rotation_90": 1.0,
        "prob_translate": 1.0,

        # ── intensity ops ─────────────────────────────────────────────────
        "prob_contrast": 1.0,
        "contrast_range": (0.8, 1.2),
        "brightness_range": (-0.08, 0.08),
        "gamma_log2_range": (-0.2, 0.2),
        "prob_gaussian_noise": 1.0,
        "gaussian_noise_std": 0.05,
        "prob_section_intensity_shift": 1.0,
        "section_intensity_shift_std": 0.05,
        "prob_block": 0.5,
        "block_shift": 10,

        # ── artifact ops ──────────────────────────────────────────────────
        "prob_darkline": 1.0,
        "darkline_width_range": (1, 12),
        "prob_missing_section": 1.0,
        "missing_section_max": 1,
        "prob_elastic": 1.0,
        "elastic_alpha": 8.0,
        "elastic_sigma": 8.0,
    }
