import torch
import torch.nn as nn
import torch.nn.functional as F

from network.network_module import *

# ----------------------------------------
#                 Generator
# ----------------------------------------
class Path1(nn.Module):
    def __init__(self, opt):
        super(Path1, self).__init__()
        self.conv1 = Conv2dLayer(opt.in_channels, 16, 3, 1, 1, pad_type = opt.pad, norm = 'none')
        self.conv2 = Conv2dLayer(16, 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.conv3 = Conv2dLayer(32, 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.conv4 = Conv2dLayer(32, 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.conv5 = Conv2dLayer(64, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')

    def forward(self, x):
        
        x1 = self.conv1(x)                                      # out: batch * 16 * 256 * 256
        x2 = self.conv2(x1)                                     # out: batch * 16 * 256 * 256
        x3 = torch.cat((x1, x2), 1)                             # out: batch * 32 * 256 * 256
        x4 = self.conv3(x3)                                     # out: batch * 16 * 256 * 256
        x5 = torch.cat((x2, x4), 1)                             # out: batch * 32 * 256 * 256
        x6 = self.conv4(x5)                                     # out: batch * 16 * 256 * 256
        x7 = torch.cat((x1, x2, x4, x6), 1)                     # out: batch * 64 * 256 * 256
        x7 = self.conv5(x7)                                     # out: batch * 31 * 256 * 256

        return x7
        
class Path2(nn.Module):
    def __init__(self, opt):
        super(Path2, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2dLayer(opt.in_channels, 8, 3, 1, 1, pad_type = opt.pad, norm = 'none'),
            DenseConv2dLayer_4C(8, 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            DenseConv2dLayer_4C(8, 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.conv2 = nn.Sequential(
            Conv2dLayer(8, 16, 3, 2, 1, pad_type = opt.pad, norm = opt.norm),
            DenseConv2dLayer_4C(16, 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            DenseConv2dLayer_4C(16, 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.conv3 = nn.Sequential(
            Conv2dLayer(16, 32, 3, 2, 1, pad_type = opt.pad, norm = opt.norm),
            DenseConv2dLayer_4C(32, 32, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            DenseConv2dLayer_4C(32, 32, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.conv4 = nn.Sequential(
            TransposeConv2dLayer(32, 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            DenseConv2dLayer_4C(16, 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            DenseConv2dLayer_4C(16, 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.conv5 = nn.Sequential(
            TransposeConv2dLayer(32, 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            DenseConv2dLayer_4C(8, 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            DenseConv2dLayer_4C(8, 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.conv6 = Conv2dLayer(16, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')

    def forward(self, x):
        
        x1 = self.conv1(x)                                      # out: batch * 8 * 256 * 256
        x2 = self.conv2(x1)                                     # out: batch * 16 * 128 * 128
        x3 = self.conv3(x2)                                     # out: batch * 32 * 64 * 64
        x4 = self.conv4(x3)                                     # out: batch * 16 * 128 * 128
        x4 = torch.cat((x4, x2), 1)                             # out: batch * 32 * 128 * 128
        x5 = self.conv5(x4)                                     # out: batch * 8 * 256 * 256
        x5 = torch.cat((x5, x1), 1)                             # out: batch * 16 * 256 * 256
        x5 = self.conv6(x5)                                     # out: batch * 31 * 256 * 256

        return x5
        
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.conv1 = Path1(opt)
        self.conv2 = Path2(opt)

    def forward(self, x):
        
        p1 = self.conv1(x)                                      # out: batch * 31 * 256 * 256
        p2 = self.conv2(x)                                      # out: batch * 31 * 256 * 256
        out = 0.5 * (p1 + p2)                                   # out: batch * 31 * 256 * 256

        return out
        
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 31, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for generator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    opt = parser.parse_args()

    a = torch.randn(1, 3, 256, 256)
    net = Generator(opt)
    b = net(a)
    print(b.shape)
