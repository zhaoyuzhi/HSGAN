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
        # The generator is U shaped
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 1, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 32 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 64 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 256 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 128 * 32 * 32
        # Decode the center code
        D1 = self.D1(E4)                                        # out: batch * 128 * 64 * 64
        D1 = torch.cat((D1, E3), 1)                             # out: batch * 256 * 64 * 64
        D2 = self.D2(D1)                                        # out: batch * 64 * 128 * 128
        D2 = torch.cat((D2, E2), 1)                             # out: batch * 128 * 128 * 128
        D3 = self.D3(D2)                                        # out: batch * 32 * 256 * 256
        D3 = torch.cat((D3, E1), 1)                             # out: batch * 64 * 256 * 256
        x = self.D4(D3)                                         # out: batch * out_channel * 256 * 256

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
