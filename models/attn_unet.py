import torch
import torch.nn as nn
from torch.distributions import Normal


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out


class AttnUNet(nn.Module):
    def __init__(self, input_steps: int, forecast_steps: int, add_noise: bool = False):
        super().__init__()
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.add_noise = add_noise

        # Encoder
        self.downsampling = nn.MaxPool2d(2, 2)
        self.in_conv = nn.Conv2d(input_steps, 32, kernel_size=1)
        self.conv1 = DoubleConv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(256, 256, kernel_size=3, padding=1)

        # Decoder
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        if self.add_noise:
            self.deconv4 = DoubleConv2d(768, 128, kernel_size=3, padding=1)
        else:
            self.deconv4 = DoubleConv2d(512, 128, kernel_size=3, padding=1)
        self.deconv3 = DoubleConv2d(256, 64, kernel_size=3, padding=1)
        self.deconv2 = DoubleConv2d(128, 32, kernel_size=3, padding=1)
        self.deconv1 = DoubleConv2d(64, 32, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(32, forecast_steps, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, C, H, W)
        batch_size, length, channels, height, width = x.size()
        h = x.reshape(batch_size, length * channels, height, width)

        # Encoding step
        h1 = self.in_conv(h)
        h2 = self.conv1(self.downsampling(h1))
        h3 = self.conv2(self.downsampling(h2))
        h4 = self.conv3(self.downsampling(h3))
        h5 = self.conv4(self.downsampling(h4))

        # Decoding step
        if self.add_noise:
            z = Normal(0.5, 2).sample(h5.size()).type_as(h5)
            h5 = torch.cat([h5, z], dim=1)
        h4p = self.deconv4(torch.cat([self.upsampling(h5), h4], dim=1))
        h3p = self.deconv3(torch.cat([self.upsampling(h4p), h3], dim=1))
        h2p = self.deconv2(torch.cat([self.upsampling(h3p), h2], dim=1))
        h1p = self.deconv1(torch.cat([self.upsampling(h2p), h1], dim=1))
        out = self.out_conv(h1p)
        out = out.reshape(batch_size, -1, channels, height, width)
        out = self.relu(out)
        return out
