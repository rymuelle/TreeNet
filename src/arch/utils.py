import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2dAdjusted(nn.Module):
    '''From NAFNet'''
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x, target_mu, target_var):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)

        y = (x - mu) / torch.sqrt(var + self.eps)

        y = y * torch.sqrt(target_var + self.eps) + target_mu

        weight_view = self.weight.view(1, self.weight.size(0), 1, 1)
        bias_view = self.bias.view(1, self.bias.size(0), 1, 1)

        y = weight_view * y + bias_view
        return y

class LayerNorm2d(nn.Module):
    '''From NAFNet'''
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)

        y = (x - mu) / torch.sqrt(var + self.eps)

        weight_view = self.weight.view(1, self.weight.size(0), 1, 1)
        bias_view = self.bias.view(1, self.bias.size(0), 1, 1)

        y = weight_view * y + bias_view
        return y

