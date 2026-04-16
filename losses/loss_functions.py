import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.float()
        sigmoid_inputs = torch.sigmoid(inputs)

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = torch.where(targets == 1, sigmoid_inputs, 1 - sigmoid_inputs)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * bce
        return focal_loss.mean()


def dice_loss(inputs, targets, smooth=1e-6):
    """Compute Dice loss (1 - Dice coefficient) using logits as input."""
    inputs = torch.sigmoid(inputs)
    targets = targets.float()
    inputs_flat = inputs.contiguous().view(inputs.shape[0], -1)
    targets_flat = targets.contiguous().view(targets.shape[0], -1)

    intersection = (inputs_flat * targets_flat).sum(dim=1)
    union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


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


def build_criterion(loss_type, dice_weight, focal_weight):
    """Build loss function from LOSS_TYPE."""
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
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
