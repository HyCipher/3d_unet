import torch


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
