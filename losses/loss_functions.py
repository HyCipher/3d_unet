import torch
import torch.nn as nn

from .focal import FocalLoss
from .dice_focal import DiceFocalLoss


def build_criterion(loss_type, dice_weight, focal_weight, pos_weight=None):
    """Build loss function from LOSS_TYPE."""
    if loss_type == "bce":
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if loss_type == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    if loss_type == "dicefocal":
        return DiceFocalLoss(
            alpha=0.25,
            gamma=2.0,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
        )
    raise ValueError(f"Unknown LOSS_TYPE: {loss_type}")
