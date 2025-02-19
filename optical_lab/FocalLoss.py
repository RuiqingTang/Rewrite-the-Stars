import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss1(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLoss2(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        loss = -(1 - inputs) ** self.gamma * targets * torch.log(inputs) - inputs ** self.gamma * (1 - targets) * torch.log(1 - inputs)
        loss = loss.mean()
        return loss