import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal import FocalLoss
from .dice import dice_loss


class DiceFocalLoss(nn.Module):
    """Combination of Dice loss and Focal loss."""

    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, inputs, targets):
        d_loss = dice_loss(inputs, targets)
        f_loss = self.focal(inputs, targets)
        return self.dice_weight * d_loss + self.focal_weight * f_loss