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