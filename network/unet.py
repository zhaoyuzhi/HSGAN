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
        # It means: input -> downsample -> upsample -> output
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E5 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E6 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E7 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E8 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # final concatenate
        self.E9 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, norm = 'none')
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D5 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D6 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D7 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D8 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D9 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'none')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 32 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 64 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 128 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 256 * 32 * 32
        E5 = self.E5(E4)                                        # out: batch * 256 * 16 * 16
        E6 = self.E6(E5)                                        # out: batch * 256 * 8 * 8
        E7 = self.E7(E6)                                        # out: batch * 256 * 4 * 4
        E8 = self.E8(E7)                                        # out: batch * 256 * 2 * 2
        # final encoding
        E9 = self.E9(E8)                                        # out: batch * 256 * 1 * 1
        # Decode the center code
        D1 = self.D1(E9)                                        # out: batch * 256 * 2 * 2
        D1 = torch.cat((D1, E8), 1)                             # out: batch * 512 * 2 * 2
        D2 = self.D2(D1)                                        # out: batch * 512 * 4 * 4
        D2 = torch.cat((D2, E7), 1)                             # out: batch * 512 * 4 * 4
        D3 = self.D3(D2)                                        # out: batch * 256 * 8 * 8
        D3 = torch.cat((D3, E6), 1)                             # out: batch * 512 * 8 * 8
        D4 = self.D4(D3)                                        # out: batch * 256 * 16 * 16
        D4 = torch.cat((D4, E5), 1)                             # out: batch * 512 * 16 * 16
        D5 = self.D5(D4)                                        # out: batch * 256 * 32 * 32
        D5 = torch.cat((D5, E4), 1)                             # out: batch * 512 * 32 * 32
        D6 = self.D6(D5)                                        # out: batch * 128 * 64 * 64
        D6 = torch.cat((D6, E3), 1)                             # out: batch * 256 * 64 * 64
        D7 = self.D7(D6)                                        # out: batch * 64 * 128 * 128
        D7 = torch.cat((D7, E2), 1)                             # out: batch * 128 * 128 * 128
        D8 = self.D8(D7)                                        # out: batch * 32 * 256 * 256
        D8 = torch.cat((D8, E1), 1)                             # out: batch * 64 * 256 * 256
        # final decoding
        x = self.D9(D8)                                         # out: batch * out_channel * 256 * 256

        return x
        