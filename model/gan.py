from typing import Any
import torch
import torch.nn as nn
from model import *


class GAN(nn.Module):
    r"""Deep Generative Adversarial Network.

    Args:
        args (args): Necessary arguments.
    """

    def __init__(self, generator: nn.Module, args: Any):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = Discriminator(
            total_steps=args.input_steps + args.forecast_steps
        )
        
        self.args = args
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class Discriminator(nn.Module):
    r"""Discriminator.

    Args:
        in_channels (list): Number of input channels.
        num_picks (int): Number of images for random selection.
    """

    def __init__(self, total_steps: int):
        super(Discriminator, self).__init__()
        self.d1 = nn.Conv2d(total_steps, 32, kernel_size=1)         # (256, 256)
        self.d2 = DoubleConv2d(32, 64, kernel_size=3, padding=1)    # (128, 128)
        self.d3 = DoubleConv2d(64, 64, kernel_size=3, padding=1)    # (64, 64)
        self.d4 = DoubleConv2d(64, 128, kernel_size=3, padding=1)   # (32, 32)
        self.d5 = DoubleConv2d(128, 128, kernel_size=3, padding=1)  # (16, 16)
        self.d6 = DoubleConv2d(128, 256, kernel_size=3, padding=1)  # (8, 8)
        self.d7 = DoubleConv2d(256, 256, kernel_size=3, padding=1)  # (4, 4)
        self.last = nn.Conv2d(256, 1, kernel_size=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed L into channel dimension
        batch_size, length, channels, height, width = x.size()
        # (B, L, C, H, W) -> (B, C*L, H, W)
        h = x.reshape(batch_size, length * channels, height, width)
        h = self.d1(h)
        h = self.d2(h)
        h = self.d3(h)
        h = self.d4(h)
        h = self.d5(h)
        h = self.d6(h)
        h = self.d7(h)
        out = self.last(h)
        # out: (B, 1)
        out = out.reshape(batch_size, 1)
        out = self.sigmoid(out)
        return out


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attn = CBAM(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.attn(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out
