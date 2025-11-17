import torch
import torch.nn as nn
from .utils import *


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ConditionedChannelAttention(nn.Module):
    def __init__(self, dims, cat_dims, out_dim=0):
        super().__init__()
        in_dim = dims + cat_dims
        if not out_dim:
            out_dim = dims
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, conditioning):
        pool = self.pool(x)
        conditioning = conditioning.unsqueeze(-1).unsqueeze(-1)
        cat_channels = torch.cat([pool, conditioning], dim=1)
        cat_channels = cat_channels.permute(0, 2, 3, 1)
        ca = self.mlp(cat_channels).permute(0, 3, 1, 2)

        return ca

class GatedConv2d(nn.Module):
    def __init__(self, channels, *args,  **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.conv = nn.Conv2d(channels, channels, *args, **kwargs)
        self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        self.gate = nn.Sigmoid()

    def forward(self, inp):
      x = self.conv(inp)
      g = self.sca(inp)
      c = self.gate(self.alpha) * self.gate(g)
      return (1-c) * inp +  c * x

# class BranchMinimal(nn.Module):
#     def __init__(self, c, kernel_size, expand=2 ):
#             super().__init__()
#             channels = c * expand
#             self.norm = LayerNorm2d(c)
#             self.conv1 = nn.Conv2d(
#                 in_channels=c,
#                 out_channels=channels,
#                 kernel_size=1,
#                 padding=0,
#                 stride=1,
#                 groups=1,
#                 bias=True,
#             )
#             padding = kernel_size // 2
#             if kernel_size > 0:
#                 self.conv2 = GatedConv2d(
#                     channels=channels,
#                     kernel_size=kernel_size,
#                     padding=padding,
#                     stride=1,
#                     groups=channels,
#                     bias=True,
#                 )
#             else:
#                 self.conv2 =  nn.Identity()

#             # Simplified Channel Attention
#             self.sca = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(in_channels=channels // 2, out_channels=channels  // 2, kernel_size=1, padding=0, stride=1,
#                           groups=1, bias=True),
#                 nn.Sigmoid()
#             )

#             # SimpleGate
#             self.sg = SimpleGate()
#             self.alpha = nn.Parameter(torch.zeros(1, c, 1, 1))

#     def forward(self, inp):
#         x = self.norm(inp)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.sg(x)
#         x = self.sca(x) * x

#         return inp + self.alpha * x


class SCA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True)
        )
    def forward(self, x):
        return x * self.sca(x)
    
class Branch(nn.Module):
    def __init__(self, c, kernel_size, expand=2 ):
            super().__init__()
            channels = c * expand
            self.norm = LayerNorm2d(c)
            self.conv1 = nn.Conv2d(
                in_channels=c,
                out_channels=channels,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            )
            padding = kernel_size // 2
            if kernel_size > 0:
                self.conv2 = nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=1,
                    groups=channels,
                    bias=True,
                )
                self.sca = SCA(channels // 2)
            else:
                self.conv2 =  nn.Identity()
                self.sca = SCA(channels // 2)

            # Simplified Channel Attention
            
            self.conv3 = nn.Conv2d(
                in_channels=channels // 2,
                out_channels=c,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            )

            # SimpleGate
            self.sg = SimpleGate()
            self.alpha = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)

        return inp + self.alpha * x


class Tree(nn.Module):
    def __init__(self, c, kernels=[3, 0, 3, 0], expand=2):
        super().__init__()
        self.trunk = nn.Sequential(*[Branch(c, k, expand=expand) for k in kernels])

    def forward(self, input):
        output = self.trunk(input)
        return output


class ForestNetV1(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 kernels_down=(3, 3, 0, 0), kernels_up=[7, 5, 3, 0]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[Tree(chan, kernels=kernels_up) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[Tree(chan, kernels=kernels_down) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[Tree(chan, kernels=kernels_down) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x