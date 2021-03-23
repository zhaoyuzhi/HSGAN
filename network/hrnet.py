import torch
import torch.nn as nn
import torch.nn.functional as F

from network.network_module import *

class GlobalBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, reduction = 8):
        super(GlobalBlock, self).__init__()
        self.conv1 = Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channels // reduction, in_channels // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channels // reduction, in_channels, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # residual
        residual = x
        # Sequeeze-and-Excitation(SE)
        b, c, _, _ = x.size()
        x = self.conv1(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = self.conv2(x)
        # addition
        out = 0.1 * y + residual
        return out

# ----------------------------------------
#                 Generator
# ----------------------------------------
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # PixelShuffle
        self.pixel_unshuffle_ratio2 = PixelUnShuffle(2)
        self.pixel_unshuffle_ratio4 = PixelUnShuffle(4)
        self.pixel_unshuffle_ratio8 = PixelUnShuffle(8)
        self.pixel_shuffle_ratio2 = PixelShuffle(2)
        # Top subnetwork, K = 3
        self.top1 = Conv2dLayer(opt.in_channels * (4 ** 3), opt.start_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.top21 = ResidualDenseBlock_5C(opt.start_channels * (2 ** 3), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.top22 = GlobalBlock(opt.start_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, sn = False, reduction = 4)
        self.top3 = Conv2dLayer(opt.start_channels * (2 ** 3), opt.start_channels * (2 ** 3), 1, 1, 0, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Middle subnetwork, K = 2
        self.mid1 = Conv2dLayer(opt.in_channels * (4 ** 2), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.mid2 = Conv2dLayer(int(opt.start_channels * (2 ** 2 + 2 ** 3 / 4)), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.mid31 = ResidualDenseBlock_5C(opt.start_channels * (2 ** 2), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.mid32 = GlobalBlock(opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, sn = False, reduction = 4)
        self.mid4 = Conv2dLayer(opt.start_channels * (2 ** 2), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Bottom subnetwork, K = 1
        self.bot1 = Conv2dLayer(opt.in_channels * (4 ** 1), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.bot2 = Conv2dLayer(int(opt.start_channels * (2 ** 1 + 2 ** 2 / 4)), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.bot31 = ResidualDenseBlock_5C(opt.start_channels * (2 ** 1), opt.start_channels * (2 ** 0), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.bot32 = ResidualDenseBlock_5C(opt.start_channels * (2 ** 1), opt.start_channels * (2 ** 0), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.bot33 = GlobalBlock(opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, sn = False, reduction = 4)
        self.bot4 = Conv2dLayer(opt.start_channels * (2 ** 1), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Mainstream
        self.main1 = Conv2dLayer(opt.in_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main2 = Conv2dLayer(int(opt.start_channels * (2 ** 0 + 2 ** 1 / 4)), opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main31 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main32 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main33 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main34 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main35 = GlobalBlock(opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, sn = False, reduction = 4)
        self.main4 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)

    def forward(self, x):
        # PixelUnShuffle                                        input: batch * 3 * 256 * 256
        x1 = self.pixel_unshuffle_ratio2(x)                     # out: batch * 12 * 128 * 128
        x2 = self.pixel_unshuffle_ratio4(x)                     # out: batch * 48 * 64 * 64
        x3 = self.pixel_unshuffle_ratio8(x)                     # out: batch * 192 * 32 * 32
        # Top subnetwork                                        suppose the start_channels = 32
        x3 = self.top1(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top21(x3)                                     # out: batch * 256 * 32 * 32
        x3 = self.top22(x3)                                     # out: batch * 256 * 32 * 32
        x3 = self.top3(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.pixel_shuffle_ratio2(x3)                      # out: batch * 64 * 64 * 64, ready to be concatenated
        # Middle subnetwork
        x2 = self.mid1(x2)                                      # out: batch * 128 * 64 * 64
        x2 = torch.cat((x2, x3), 1)                             # out: batch * (128 + 64) * 64 * 64
        x2 = self.mid2(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid31(x2)                                     # out: batch * 128 * 64 * 64
        x2 = self.mid32(x2)                                     # out: batch * 128 * 64 * 64
        x2 = self.mid4(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.pixel_shuffle_ratio2(x2)                      # out: batch * 32 * 128 * 128, ready to be concatenated
        # Bottom subnetwork
        x1 = self.bot1(x1)                                      # out: batch * 64 * 128 * 128
        x1 = torch.cat((x1, x2), 1)                             # out: batch * (64 + 32) * 128 * 128
        x1 = self.bot2(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot31(x1)                                     # out: batch * 64 * 128 * 128
        x1 = self.bot32(x1)                                     # out: batch * 64 * 128 * 128
        x1 = self.bot33(x1)                                     # out: batch * 64 * 128 * 128
        x1 = self.bot4(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.pixel_shuffle_ratio2(x1)                      # out: batch * 16 * 256 * 256, ready to be concatenated
        # U-Net generator with skip connections from encoder to decoder
        x = self.main1(x)                                       # out: batch * 32 * 256 * 256
        x = torch.cat((x, x1), 1)                               # out: batch * (32 + 16) * 256 * 256
        x = self.main2(x)                                       # out: batch * 32 * 256 * 256
        x = self.main31(x)                                      # out: batch * 32 * 256 * 256
        x = self.main32(x)                                      # out: batch * 32 * 256 * 256
        x = self.main33(x)                                      # out: batch * 32 * 256 * 256
        x = self.main34(x)                                      # out: batch * 32 * 256 * 256
        x = self.main35(x)                                      # out: batch * 32 * 256 * 256
        x = self.main4(x)                                       # out: batch * 3 * 256 * 256

        return x

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 31, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for generator')
    parser.add_argument('--latent_channels', type = int, default = 16, help = 'start channels for generator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    opt = parser.parse_args()

    a = torch.randn(1, 3, 256, 256)
    net = Generator(opt)
    b = net(a)
    print(b.shape)
