import torch
import torch.nn as nn
from torch.distributions import Normal
from .layers import DoubleConv2d, DoubleDeconv2d


class AttnUNet(nn.Module):
    def __init__(self, input_steps: int, forecast_steps: int, add_noise: bool = False):
        super(AttnUNet, self).__init__()
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.add_noise = add_noise

        # Encoder
        self.in_conv = nn.Conv2d(input_steps, 32, kernel_size=1)
        self.down1 = DoubleConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = DoubleConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.down3 = DoubleConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.down4 = DoubleConv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # Dilation convolutions
        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder
        if self.add_noise:
            self.up4 = DoubleDeconv2d(1536, 256, kernel_size=4, stride=2, padding=1)
        else:
            self.up4 = DoubleDeconv2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.up3 = DoubleDeconv2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.up2 = DoubleDeconv2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.up1 = DoubleDeconv2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(32, forecast_steps, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, C, H, W)
        batch_size, length, channels, height, width = x.size()
        h = x.reshape(batch_size, length * channels, height, width)

        # Encoding step
        h1 = self.in_conv(h)
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h4 = self.down3(h3)
        h5 = self.down4(h4)

        # Dilation step
        hd = self.dilated_conv1(h5)
        hd = self.dilated_conv2(hd)
        hd = self.dilated_conv3(hd)
        hd = self.dilated_conv4(hd)

        # Decoding step
        if self.add_noise:
            z = Normal(0, 1).sample(hd.size()).type_as(hd)
            hd = torch.cat([hd, z], dim=1)
        h4p = self.up4(torch.cat([hd, h5], dim=1))
        h3p = self.up3(torch.cat([h4p, h4], dim=1))
        h2p = self.up2(torch.cat([h3p, h3], dim=1))
        h1p = self.up1(torch.cat([h2p, h2], dim=1))
        out = self.out_conv(h1p)
        out = out.reshape(batch_size, -1, channels, height, width)
        out = self.relu(out)
        return out
        