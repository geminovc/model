import torch
from torch import nn
from torch.nn import Conv2d
import torch.nn.functional as F


USE_FAST_CONV2 = False


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        if not USE_FAST_CONV2:
            self.conv1 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                padding=padding)
            self.conv2 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                            padding=padding)
        else:
            self.conv1 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.point_conv1 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=(1, 1))
            self.conv2 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.point_conv2 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=(1, 1))
        self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features, affine=True)
        self.relu = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        out = self.norm1(x)
        out = self.relu(out)
        if not USE_FAST_CONV2:
            out = self.conv1(out)
        else:
            out = self.conv1(out)
            out = self.point_conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        if not USE_FAST_CONV2:
            out = self.conv2(out)
        else:
            out = self.conv2(out)
            out = self.point_conv2(out)
        out = self.dequant(out)
        x = self.dequant(x)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        if not USE_FAST_CONV2:
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                padding=padding, groups=groups)
        else:
            self.depth_conv = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=(1, 1))
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.relu = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out = F.interpolate(x, scale_factor=2)
        if not USE_FAST_CONV2:
            out = self.conv(out)
        else:
            out = self.depth_conv(out)
            out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dequant(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        if not USE_FAST_CONV2:
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                padding=padding, groups=groups)
        else:
            self.depth_conv = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=(1, 1))
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.relu = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        if not USE_FAST_CONV2:
            out = self.conv(x)
        else:
            out = self.depth_conv(x)
            out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dequant(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        if not USE_FAST_CONV2:
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features,
                                kernel_size=kernel_size, padding=padding, groups=groups)
        else:
            self.depth_conv = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=(1, 1))
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.relu = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        if not USE_FAST_CONV2:
            out = self.conv(x)
        else:
            out = self.depth_conv(x)
            out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dequant(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        # x = self.quant(x)
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        # outs = self.dequant(outs)
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()
        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

