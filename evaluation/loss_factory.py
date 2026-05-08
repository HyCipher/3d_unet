import torch.nn as nn

from losses import DiceFocalLoss, FocalLoss


def build_validation_criterion(loss_type):
    """Build validation criterion from configured loss type."""
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_type == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    if loss_type == "dicefocal":
        return DiceFocalLoss(alpha=0.25, gamma=2.0, dice_weight=0.8, focal_weight=1.5)
    return None