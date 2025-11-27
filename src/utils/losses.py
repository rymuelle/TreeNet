import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class PSNR_LCS(nn.Module):
    def __init__(self, kernel_size=11, weights=[1,1]
                 ):
        super().__init__()
        self.psnr_loss = PSNRLoss()
        self.lcs = LocalCosineSimilarity(kernel_size=kernel_size)
        self.weights = weights
        
    def forward(self, pred, target):
        loss = 0
        if self.weights[0] > 0: 
            loss += self.weights[0] * self.psnr_loss(pred, target) 
        if self.weights[1] > 0: 
            loss += self.weights[1] * (1-self.lcs(pred, target)).mean()
        return loss


import torch.nn as nn

class LocalCosineSimilarity(nn.Module):
    def __init__(self, kernel_size=11, sigma=1, eps=1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.constant = 2 * math.pi * math.e
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold((kernel_size, kernel_size), dilation=1, padding=padding, stride=1)
        self.eps = eps
        

    def forward(self, x, y):
        B, C, H, W = x.shape
        K = self.kernel_size

        # B C K K H W
        x_patches = self.unfold(x)
        x_patches = x_patches.view(B, C, K, K, H, W)
        y_patches = self.unfold(y)
        y_patches = y_patches.view(B, C, K, K, H, W)

        # B C H W
        x_norm = (x_patches ** 2).sum(dim=(1, 2, 3)) ** .5
        y_norm = (y_patches ** 2).sum(dim=(1, 2, 3)) ** .5

        # B C H W
        dot = (x_patches * y_patches).sum(dim=(1, 2, 3))
        cosine_similarity = dot/(x_norm*y_norm + self.eps)

        return cosine_similarity

    def __repr__(self):
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, eps={self.eps})'

class PixelPSNRLoss(nn.Module):
    '''From NAFNet'''
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PixelPSNRLoss, self).__init__()
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

        return self.loss_weight * self.scale * torch.log((pred - target) ** 2 + 1e-6).mean()

    def __repr__(self):
      return f'{self.__class__.__name__}(loss_weight={self.loss_weight}, toY={self.toY})'

class PPSNR_LCS(nn.Module):
    def __init__(self, kernel_size=11, weights=[1,1], eps=1
                 ):
        super().__init__()
        self.psnr_loss = PixelPSNRLoss()
        self.lcs = LocalCosineSimilarity(kernel_size=kernel_size, eps=eps)
        self.weights = weights

    def forward(self, pred, target):
        loss = 0
        if self.weights[0] > 0:
            loss += self.weights[0] * self.psnr_loss(pred, target)
        if self.weights[1] > 0:
            loss += self.weights[1] * (1-self.lcs(pred, target)).mean()
        return loss

    def __repr__(self):
        return f'{self.__class__.__name__}(psnr_loss={self.psnr_loss}, lcs={self.lcs}, weights={self.weights})'
    
class WeightedLocalCosineSimilarity(nn.Module):
    def __init__(self, kernel_size=11, sigma=None, eps=1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.constant = 2 * math.pi * math.e
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold((kernel_size, kernel_size), dilation=1, padding=padding, stride=1)
        self.eps = eps

        # Gaussian Kernel
        if sigma is None:
            sigma = kernel_size
        coords = torch.arange(kernel_size) - padding
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        g = g / g.sum()      
        self.register_buffer("gauss", g)  


    def forward(self, x, y):
        B, C, H, W = x.shape
        K = self.kernel_size

        # B C K K H W
        x_patches = self.unfold(x)
        x_patches = x_patches.view(B, C, K, K, H, W)
        y_patches = self.unfold(y)
        y_patches = y_patches.view(B, C, K, K, H, W)

        # 1 1 K K 1 1
        w = self.gauss.view(1, 1, K, K, 1, 1)
        
        # B C H W
        x_norm = (w * (x_patches ** 2) ).sum(dim=(1, 2, 3)) ** .5
        y_norm = (w * (y_patches ** 2) ).sum(dim=(1, 2, 3)) ** .5

        # B C H W
        dot = (w * x_patches * y_patches).sum(dim=(1, 2, 3))
        cosine_similarity = dot/(x_norm*y_norm + self.eps)

        return cosine_similarity
    
class LocalSTD(nn.Module):
    def __init__(self, kernel_size=11, sigma=1, stabilization=1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.constant = 2 * math.pi * math.e
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold((kernel_size, kernel_size), dilation=1, padding=padding, stride=1)
        self.stabilization = stabilization
        
        # Gaussian Kernel
        if sigma is None:
            sigma = kernel_size / 3.0  
        coords = torch.arange(kernel_size) - padding
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        g = g / g.sum()      # normalize to sum=1
        self.register_buffer("gauss", g)   # shape (K, K)

    def forward(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size

        patches = self.unfold(x)
        patches = patches.view(B, C, K, K, H, W)

        w = self.gauss.view(1, 1, K, K, 1, 1)

        mean = (patches * w).sum(dim=(2, 3), keepdim=True)
        var = (w * (patches - mean)**2).sum(dim=(2, 3)) + self.stabilization

        return var ** .5
    

class LocalEntropy(nn.Module):
    def __init__(self, kernel_size=11, sigma=1, stabilization=1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.constant = 2 * math.pi * math.e
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold((kernel_size, kernel_size), dilation=1, padding=padding, stride=1)
        self.stabilization = stabilization
        
        # Gaussian Kernel
        if sigma is None:
            sigma = kernel_size / 3.0  
        coords = torch.arange(kernel_size) - padding
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        g = g / g.sum()      # normalize to sum=1
        self.register_buffer("gauss", g)   # shape (K, K)

    def forward(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size

        patches = self.unfold(x)
        patches = patches.view(B, C, K, K, H, W)

        w = self.gauss.view(1, 1, K, K, 1, 1)

        mean = (patches * w).sum(dim=(2, 3), keepdim=True)
        var = (w * (patches - mean)**2).sum(dim=(2, 3)) + self.stabilization

        entropy = 0.5 * torch.log(self.constant * var)
        return entropy
    

class LocalEntropyDW(nn.Module):
    def __init__(self, kernel_size=5, stabilization=1e-6, sigma=None):
        super().__init__()
        padding = kernel_size // 2
        self.constant = 2 * math.pi * math.e
        self.stabilization = stabilization
        
        if sigma is None:
            sigma = kernel_size / 3.0
        
        coords = torch.arange(kernel_size) - padding
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        g = g / g.sum()
        
        # conv weight shape: (C, 1, K, K), but C unknown until forward
        self.register_buffer("gauss", g[None, None, :, :])
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Repeat Gaussian for depthwise convolution
        # Shape needed: (C, 1, K, K)
        weight = self.gauss.expand(C, 1, self.kernel_size, self.kernel_size)
        
        # ---- Weighted mean ----
        mean = F.conv2d(x, weight, padding=self.padding, groups=C)

        # ---- Weighted variance ----
        var = F.conv2d((x - mean)**2, weight, padding=self.padding, groups=C)
        var = var + self.stabilization

        # ---- Entropy map ----
        entropy = 0.5 * torch.log(self.constant * var)
        return entropy




class LocalEntropyLoss(nn.Module):
    def __init__(self, kernel_size=5, criterion=nn.MSELoss):
        super().__init__()
        self.le = LocalEntropyDW(kernel_size=kernel_size)
        self.criterion = criterion()

    def forward(self, pred, target):
        ep = self.le(pred)
        et = self.le(target)
        loss = self.criterion(ep, et)
        return loss
    

class PSNR_LE_loss(nn.Module):
    def __init__(self, kernel_size=5, criterion=nn.MSELoss, weights=[1, 1e1]):
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




def custom_weight_init(m):
    if isinstance(m, nn.Conv2d):
        k = m.kernel_size[0]  # assuming square kernels
        c_in = m.in_channels
        n_l = (k ** 2) * c_in
        std = math.sqrt(2.0 / n_l)
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        n_l = m.in_features
        std = math.sqrt(2.0 / n_l)
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class VGGFeatureExtractor(nn.Module):
    """
    VGG-like network for perceptual loss with runtime hot-swappable activations.
    Returns features from selected layers.
    """
    def __init__(self, config=None, feature_layers=None, activation=None):
        super().__init__()
        if config is None:
            config = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
        
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22, 29]

        if activation is None:
            activation = lambda: nn.ReLU(inplace=False)
        
        layers = []
        in_channels = 3
        for num_convs, out_channels in config:
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(activation())
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.features = nn.Sequential(*layers)
        self.feature_layers = feature_layers
        self.activation_factory = activation  # store for later swapping
        self.apply(custom_weight_init)

    def set_activation(self, activation_cls, **kwargs):
        """
        Replace all activation layers with new activation.
        activation_cls: activation class (e.g., nn.GELU, nn.LeakyReLU)
        kwargs: any keyword args for the activation class
        """
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.ReLU) or \
               isinstance(layer, nn.LeakyReLU) or \
               isinstance(layer, nn.GELU) or \
               isinstance(layer, nn.SiLU) or \
               isinstance(layer, nn.Identity) or \
               isinstance(layer, nn.Tanh):
                self.features[i] = activation_cls(**kwargs)
        # Update the stored factory for future reference
        self.activation_factory = lambda: activation_cls(**kwargs)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                outputs.append(x)
        return outputs




class PSNR_LE_VFE_loss(nn.Module):
    def __init__(self, kernel_size=5, criterion=nn.MSELoss, weights=[1, 1e1, 1],
                 vfe_config=[(1, 64), (1, 128), (1, 256), (1, 512), (1, 512)],
                 feature_layers=[1, 14]
                 ):
        super().__init__()
        self.psnr_loss = PSNRLoss()
        self.le_loss = LocalEntropyLoss(kernel_size=kernel_size, criterion=criterion)
        self.weights = weights
        self.vfe = VGGFeatureExtractor(config=vfe_config, feature_layers=feature_layers)
        self.vfw_loss_fn = nn.MSELoss()

    def forward(self, pred, target):
        loss = 0
        if self.weights[0] > 0: 
            loss += self.psnr_loss(pred, target)
        if self.weights[1] > 0: 
            loss += self.le_loss(pred, target) * self.weights[1]
        if self.weights[2] > 0:
            xps = self.vfe(pred)
            xts = self.vfe(target)
            for xp, xt in zip(xps, xts):
                loss += self.vfw_loss_fn(xp, xt) * self.weights[2]
        return loss


class L1_LE_VFE_loss(nn.Module):
    def __init__(self, kernel_size=5, criterion=nn.MSELoss, weights=[1, .1, 20],
                 vfe_config=[(1, 64), (1, 128), (1, 256), (1, 512), (1, 512)],
                 feature_layers=[1, 14],
                 reroll_weights = True,
                 ):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.le_loss = LocalEntropyLoss(kernel_size=kernel_size, criterion=criterion)
        self.weights = weights
        self.vfe = VGGFeatureExtractor(config=vfe_config, feature_layers=feature_layers)
        self.vfw_loss_fn = nn.MSELoss()
        self.reroll_weights = reroll_weights

    def forward(self, pred, target):
        loss = 0
        if self.weights[0] > 0: 
            loss += self.l1_loss(pred, target)
        if self.weights[1] > 0: 
            loss += self.le_loss(pred, target) * self.weights[1]
        if self.weights[2] > 0:
            xps = self.vfe(pred)
            xts = self.vfe(target)
            for xp, xt in zip(xps, xts):
                loss += self.vfw_loss_fn(xp, xt) * self.weights[2]
            if self.reroll_weights:
                self.vfe.apply(custom_weight_init)
        return loss


class PSNR_LE_VFE_loss(nn.Module):
    def __init__(self, kernel_size=5, criterion=nn.MSELoss, weights=[1, .1, 20],
                 vfe_config=[(1, 64), (1, 128), (1, 256), (1, 512), (1, 512)],
                 feature_layers=[1, 14],
                 reroll_weights = True,
                 ):
        super().__init__()
        self.le = LocalEntropy(kernel_size=kernel_size)
        self.l1_loss = nn.L1Loss()
        self.le_loss = LocalEntropyLoss(kernel_size=kernel_size, criterion=criterion)
        self.weights = weights
        self.vfe = VGGFeatureExtractor(config=vfe_config, feature_layers=feature_layers)
        self.vfw_loss_fn = nn.MSELoss()
        self.reroll_weights = reroll_weights

    def forward(self, pred, target):
        loss = 0
        le = LocalEntropy
        if self.weights[0] > 0: 
            loss += PSNRLoss()
        if self.weights[1] > 0: 
            loss += self.le_loss(pred, target) * self.weights[1]
        if self.weights[2] > 0:
            xps = self.vfe(pred)
            xts = self.vfe(target)
            for xp, xt in zip(xps, xts):
                loss += self.vfw_loss_fn(xp, xt) * self.weights[2]
            if self.reroll_weights:
                self.vfe.apply(custom_weight_init)
        return loss
