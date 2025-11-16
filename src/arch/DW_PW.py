# ------------------------------------------------------------------------
# Modified from NAFNet
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import torch
import torch.nn as nn
from .utils import *

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class DWBlock(nn.Module):
    def __init__(self, c, DW_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.sg = SimpleGate()
        self.norm = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        return inp + x * self.beta
    

class DWBlockV2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                               bias=True)
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        self.sg = SimpleGate()
        self.norm = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        norm_x = self.norm(x)
        x = self.conv1(norm_x)
        x1 = self.conv2(x)
        x2 = self.conv3(norm_x)
        x = x1 * x2
        # x = x * self.sca(x)
        return inp + x * self.beta


class PWBlock(nn.Module):
    def __init__(self, c, FFN_Expand=2):
        super().__init__()
        ffn_channel = FFN_Expand * c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg = SimpleGate()
        self.norm = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
    def forward(self, inp):
        x = inp
        x = self.norm(x)
        x = x * self.sca(x)
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        return inp + x * self.beta


class DWPW(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=(1,1), enc_blk_nums=[(1,1)], dec_blk_nums=[(1,1)]):
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
        for dwnum, pwnum in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[DWBlockV2(chan) for _ in range(dwnum)],
                    *[PWBlock(chan) for _ in range(pwnum)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                    *[DWBlockV2(chan) for _ in range(middle_blk_num[0])],
                    *[PWBlock(chan) for _ in range(middle_blk_num[1])]
            )

        for dwnum, pwnum in dec_blk_nums[::-1]:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[DWBlockV2(chan) for _ in range(dwnum)],
                    *[PWBlock(chan) for _ in range(pwnum)]
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
