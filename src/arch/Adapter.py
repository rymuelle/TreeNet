import torch.nn as nn
import torch
from src.arch.NAFNet import NAFBlock

class Adapter(nn.Module):
    def __init__(self, rgb_multiplier=[2., 1. ,2.],
                 gamma=1./2, intro_blocks=0, outro_blocks=0, width=32):
        super().__init__()

        self.intro = nn.Sequential(nn.Conv2d(3, width, kernel_size=1),
            nn.GELU(),
            *[NAFBlock(width) for _ in range(intro_blocks)],
            nn.Conv2d(width, 3, kernel_size=1))
        
        self.outro = nn.Sequential(nn.Conv2d(3, width, kernel_size=1),
            nn.GELU(),
            *[NAFBlock(width) for _ in range(outro_blocks)],
            nn.Conv2d(width, 3, kernel_size=1))
        
        self.camera_rgb_adjust = torch.nn.Parameter(torch.tensor(rgb_multiplier).view(1, 3, 1, 1), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.tensor([gamma]), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros((1, 3, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, 3, 1, 1)), requires_grad=True)
    def to_proc(self, inp):
        # To processed like
        inp = inp ** self.gamma
        inp = inp * self.camera_rgb_adjust
        inp = self.intro(inp) * self.alpha + inp
        return inp

    def to_RAW(self, out):
        # To RAW like
        out = self.outro(out) * self.beta + out
        out = out /  self.camera_rgb_adjust
        out = torch.clamp(out, 1e-6, 1.0)

        out = out ** (1./self.gamma)
        return out
