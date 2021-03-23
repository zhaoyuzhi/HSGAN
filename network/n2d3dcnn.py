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
        self.conv = nn.Sequential(
            Conv2dLayer(opt.in_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none'),
            Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')
        )

    def forward(self, x):
        
        x = self.conv(x)                                      # out: batch * 3 * 256 * 256

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
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    opt = parser.parse_args()

    a = torch.randn(1, 3, 256, 256)
    net = Generator(opt)
    b = net(a)
    print(b.shape)
