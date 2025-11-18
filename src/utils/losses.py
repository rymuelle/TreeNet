import numpy as np
import math
import torch
import torch.nn as nn

def psnr(mse_loss):
    """Convert MSE to PSNR (dB). Assumes input images in [0,1]."""
    return -10.0 * math.log10(mse_loss) if mse_loss > 0 else 100.0

class PSNRLoss(nn.Module):
    '''From NAFNet'''
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

loss_fn = PSNRLoss()


class LocalEntropy(nn.Module):
    def __init__(self, kernel_size=5, stablization = 1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.constant = 2 * math.pi * math.e
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold((kernel_size, kernel_size), dilation=1, padding=padding, stride=1)
        self.stablization = stablization
        
    def forward(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size
        patches_flat = self.unfold(x)
        patches_reshaped = patches_flat.view(B, C, K, K, H, W)
        var = patches_reshaped.var(dim=(2, 3), keepdim=False) + self.stablization
        entropy = 1/2 * torch.log(self.constant * var)
        return entropy


class LocalEntropyLoss(nn.Module):
    def __init__(self, kernel_size=5, criterion=nn.MSELoss):
        super().__init__()
        self.le = LocalEntropy(kernel_size=kernel_size)
        self.criterion = criterion()

    def forward(self, pred, target):
        ep = self.le(pred)
        et = self.le(target)
        loss = self.criterion(ep, et)
        return loss
    

class PSNR_LE_loss(nn.Module):
    def __init__(self, kernel_size=5, criterion=nn.MSELoss, weights=[1, 1]):
        super().__init__()
        self.psnr_loss = PSNRLoss()
        self.le_loss = LocalEntropyLoss(kernel_size=kernel_size, criterion=criterion)
        self.weights = weights

    def forward(self, pred, target):
        loss = 0
        if self.weights[0] > 0: 
            loss += self.psnr_loss(pred, target)
        if self.weights[1] > 0: 
            loss += self.le_loss(pred, target) + self.weights[1]
        return loss