import torch
import torch.nn as nn
import torch.nn.functional as F

from network.network_module import *

# ----------------------------------------
#           Downsampling Block
# ----------------------------------------
class LSSR_Downsampler(nn.Module):
    def __init__(self, in_channels, kernel_size = 1, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(LSSR_Downsampler, self).__init__()
        # Initialize LSSR_Downsampler
        self.conv2d = Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.maxpool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.maxpool(x)
        return x

class LSSR_Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, scale_factor = 2):
        super(LSSR_Upsampler, self).__init__()
        # Initialize LSSR_Downsampler
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        return x

# ----------------------------------------
#                 Generator
# ----------------------------------------
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.enc1 = Conv2dLayer(opt.in_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none')
        self.enc2 = nn.Sequential(
            DenseConv2dLayer_4C(opt.start_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            LSSR_Downsampler(opt.start_channels, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        )
        self.enc3 = nn.Sequential(
            DenseConv2dLayer_4C(opt.start_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            LSSR_Downsampler(opt.start_channels, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        )
        self.enc4 = nn.Sequential(
            DenseConv2dLayer_4C(opt.start_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            LSSR_Downsampler(opt.start_channels, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        )
        self.enc5 = nn.Sequential(
            DenseConv2dLayer_4C(opt.start_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            LSSR_Downsampler(opt.start_channels, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        )
        self.bottleneck = Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.dec1 = nn.Sequential(
            DenseConv2dLayer_4C(opt.start_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            LSSR_Upsampler(opt.start_channels, opt.start_channels, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        )
        self.dec2 = nn.Sequential(
            DenseConv2dLayer_4C(opt.start_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            LSSR_Upsampler(opt.start_channels * 2, opt.start_channels, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        )
        self.dec3 = nn.Sequential(
            DenseConv2dLayer_4C(opt.start_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            LSSR_Upsampler(opt.start_channels * 2, opt.start_channels, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        )
        self.dec4 = nn.Sequential(
            DenseConv2dLayer_4C(opt.start_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            LSSR_Upsampler(opt.start_channels * 2, opt.start_channels, 1, 1, 0, pad_type = opt.pad, norm = opt.norm)
        )
        self.dec5 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')

    def forward(self, x):
        
        x1 = self.enc1(x)                                       # out: batch * 64 * 256 * 256
        x2 = self.enc2(x1)                                      # out: batch * 64 * 128 * 128
        x3 = self.enc3(x2)                                      # out: batch * 64 * 64 * 64
        x4 = self.enc4(x3)                                      # out: batch * 64 * 32 * 32
        x5 = self.enc5(x4)                                      # out: batch * 64 * 16 * 16
        x5 = self.bottleneck(x5)                                # out: batch * 64 * 16 * 16
        d1 = self.dec1(x5)                                      # out: batch * 64 * 32 * 32
        d1 = torch.cat((d1, x4), 1)                             # out: batch * 128 * 32 * 32
        d2 = self.dec2(d1)                                      # out: batch * 64 * 64 * 64
        d2 = torch.cat((d2, x3), 1)                             # out: batch * 128 * 64 * 64
        d3 = self.dec3(d2)                                      # out: batch * 64 * 128 * 128
        d3 = torch.cat((d3, x2), 1)                             # out: batch * 128 * 128 * 128
        d4 = self.dec4(d3)                                      # out: batch * 64 * 256 * 256
        d4 = torch.cat((d4, x1), 1)                             # out: batch * 128 * 256 * 256
        d5 = self.dec5(d4)                                      # out: batch * 3 * 256 * 256

        return d5

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
