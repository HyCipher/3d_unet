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
        "blackpad_prob": 0.15,

        # ── intensity ops ─────────────────────────────────────────────────
        "contrast_range": (0.8, 1.2),
        "brightness_range": (-0.08, 0.08),
        "gamma_log2_range": (-0.2, 0.2),
        "gaussian_noise_std": 0.05,
        "section_intensity_shift_std": 0.05,
        "block_prob": 0.5,
        "block_shift": 10,

        # ── artifact ops ──────────────────────────────────────────────────
        "darkline_width_range": (1, 12),
        "missing_section_max": 1,
        "elastic_alpha": 8.0,
        "elastic_sigma": 8.0,
    }
