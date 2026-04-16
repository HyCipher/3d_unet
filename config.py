def get_control_panel():
    """Centralized training/validation hyperparameters."""
    return {
        "validate_every": 10,
        "eval_train_set": True,
        "max_val_volumes": None,
        "val_patch_size": (16, 512, 512),
        "val_stride": (8, 256, 256),
        "val_threshold": 0.5,  # 优化: 从0.1改为0.5，找到更平衡的决策边界
        "dice_weight": 0.5,  # 优化: 从0.8改为0.5，Dice Loss权重更高
        "focal_weight": 1.0,
        "num_epochs": 50,
        "loss_type": "dicefocal",  # 优化: 从"bce"改为"dicefocal"，更好处理类不平衡
    }
