import torch.nn as nn
import torch
import torch.nn.functional as F

class InputNormalizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

        self.camera_rgb_adjust = torch.nn.Parameter(torch.tensor([1.8, 1., 1.8]).view(1, 3, 1, 1), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.tensor([0.6]), requires_grad=True)
    def forward(self, inp, cond):

        inp = inp ** self.gamma

        inp = inp * self.camera_rgb_adjust

        out = self.backbone(inp)

        out = out / self.camera_rgb_adjust
        out = torch.clamp(out, 0.0, 1.0)

        out = out ** (1./self.gamma)
        return out


class InputNormalizerST(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

        self.camera_rgb_adjust = torch.nn.Parameter(torch.tensor([1.8, 1., 1.8]).view(1, 3, 1, 1), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.tensor([0.6]), requires_grad=True)
    def forward(self, inp, cond):

        inp = inp ** self.gamma

        inp = inp * self.camera_rgb_adjust

        out = self.backbone(inp)

        out[-1] = out[-1] / self.camera_rgb_adjust
        out[-1] = torch.clamp(out[-1], 0.0, 1.0)

        out[-1] = out[-1] ** (1./self.gamma)
        return out
    


class MultiLevelFeatureLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels, weights=None):
        super(MultiLevelFeatureLoss, self).__init__()
        
        self.projections = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for s_ch, t_ch in zip(student_channels, teacher_channels):
            if s_ch != t_ch:
                self.projections.append(nn.Conv2d(s_ch, t_ch, kernel_size=1))
            else:
                self.projections.append(nn.Identity())
            
            self.norms.append(nn.LayerNorm(t_ch))

        self.weights = weights if weights is not None else [1.0] * len(student_channels)

    def forward(self, student_feats, teacher_feats):
        total_loss = 0
        
        for i, (s_f, t_f) in enumerate(zip(student_feats, teacher_feats)):
            s_f = self.projections[i](s_f)
            
            if s_f.shape[2:] != t_f.shape[2:]:
                s_f = F.interpolate(s_f, size=t_f.shape[2:], mode='bilinear', align_corners=False)
            
            s_f = self._apply_norm(s_f, self.norms[i])
            t_f = self._apply_norm(t_f, self.norms[i])
            
            total_loss += self.weights[i] * F.mse_loss(s_f, t_f)
            
        return total_loss

    def _apply_norm(self, x, norm_layer):
        x = x.permute(0, 2, 3, 1)
        x = norm_layer(x)
        return x.permute(0, 3, 1, 2)