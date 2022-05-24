from typing import Tuple, Union
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell without peephole connection.
        
    Args:
        channels (int): number of input channels.
        filters (int): number of convolutional kernels.
        kernel_size (int, tuple): size of convolutional kernels.
        padding (int, tuple): size of padding.
    """
    
    def __init__(self, channels: int, filters: int, kernel_size: Union[int, tuple] = 3, 
                 padding: Union[int, tuple] = 1):
        super(ConvLSTMCell, self).__init__()
        self.filters = filters
        self.conv = nn.Conv2d(channels + filters, filters * 4, kernel_size, padding=padding)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: 4D tensor (B, C, H, W)
        batch_size, _, height, width = x.size()
        # Initialize h and c with torch.zeros
        if h is None:
            h = torch.zeros(size=(batch_size, self.filters, height, width)).type_as(x)
        if c is None:
            c = torch.zeros(size=(batch_size, self.filters, height, width)).type_as(x)
        # forward process
        i, f, g, o = torch.split(self.conv(torch.cat([x, h], dim=1)), self.filters, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        
        return h, c


class Down(nn.Module):
    """Convolutional cell for 5D (S, B, C, H, W) input. The Down consists of 2 parts, 
        the ResNet bottleneck and the SENet module (optional). 
        
    Args:
        channels (int): Number of input channels.
        filters (int): Number of convolutional kernels.
        kernel_size (int, tuple): Size of convolutional kernels.
        stride (int, tuple): Stride of the convolution.
        padding (int, tuple): Padding of the convolution.
    """
    
    def __init__(self, channels: int, filters: int, kernel_size: Union[int, tuple] = 3, padding: Union[int, tuple] = 1):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(channels, filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        # self.channel = ChannelAttention(filters)
        # self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        # x = self.channel(x) * x
        # x = self.spatial(x) * x
        return x


class Up(nn.Module):
    """Bilinear-resize-convolutional cell for 5D (S, B, C, H, W) input. 
        
    Args:
        channels (int): Number of input channels.
        filters (int): Number of convolutional kernels.
        kernel_size (int, tuple): Size of convolutional kernels.
        padding (int, tuple): Padding of the convolution.
        attention (bool): Whether to add attentional module.
    """
    
    def __init__(self, channels: int, filters: int, kernel_size: Union[int, tuple] = 3, 
                 padding: Union[int, tuple] = 1, attention: bool = False):
        super(Up, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels, filters, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        self.attention = attention
        # self.channel = ChannelAttention(filters)
        # self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # x = self.channel(x) * x
        # x = self.spatial(x) * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
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

