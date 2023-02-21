import torch
import torch.nn as nn
from torch.distributions import Normal
from .layers import DoubleConv2d


class AttnUNet(nn.Module):
    def __init__(self, input_steps: int, forecast_steps: int, add_noise: bool = False):
        super(AttnUNet, self).__init__()
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
            z = Normal(0, 1).sample(h5.size()).type_as(h5)
            h5 = torch.cat([h5, z], dim=1)
        h4p = self.deconv4(torch.cat([self.upsampling(h5), h4], dim=1))
        h3p = self.deconv3(torch.cat([self.upsampling(h4p), h3], dim=1))
        h2p = self.deconv2(torch.cat([self.upsampling(h3p), h2], dim=1))
        h1p = self.deconv1(torch.cat([self.upsampling(h2p), h1], dim=1))
        out = self.out_conv(h1p)
        out = out.reshape(batch_size, -1, channels, height, width)
        out = self.relu(out)
        return out
        