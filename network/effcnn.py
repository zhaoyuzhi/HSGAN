import torch
import torch.nn as nn
import torch.nn.functional as F

from network.network_module import *

# ----------------------------------------
#                 Generator
# ----------------------------------------
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.subpath = Conv2dLayer(opt.in_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')
        self.conv1 = Conv2dLayer(opt.in_channels, 128, 5, 1, 2, pad_type = opt.pad, norm = 'none')
        self.conv2 = Conv2dLayer(128, 32, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        self.conv3 = ResConv2dLayer(32, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.conv4 = ResConv2dLayer(32, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.conv5 = Conv2dLayer(64, 128, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        self.conv6 = Conv2dLayer(256, opt.out_channels, 5, 1, 2, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')

    def forward(self, x):
        
        x1 = self.conv1(x)                                      # out: batch * 128 * 256 * 256
        x2 = self.conv2(x1)                                     # out: batch * 32 * 256 * 256
        x3 = self.conv3(x2)                                     # out: batch * 32 * 256 * 256
        x3 = self.conv4(x3)                                     # out: batch * 32 * 256 * 256
        x3 = torch.cat((x3, x2), 1)                             # out: batch * 64 * 256 * 256
        x3 = self.conv5(x3)                                     # out: batch * 128 * 256 * 256
        x3 = torch.cat((x3, x1), 1)                             # out: batch * 256 * 256 * 256
        p1 = self.conv6(x3)                                     # out: batch * 31 * 256 * 256
        p2 = self.subpath(x)                                    # out: batch * 31 * 256 * 256
        out = 0.5 * (p1 + p2)                                   # out: batch * 3 * 256 * 256

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
