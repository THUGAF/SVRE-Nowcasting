from typing import Tuple, Union, List
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):    
    def __init__(self, channels: int, filters: int, kernel_size: Union[int, tuple] = 3, 
                 padding: Union[int, tuple] = 1):
        super().__init__()
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
    def __init__(self, channels: int, filters: int, kernel_size: Union[int, tuple] = 3, padding: Union[int, tuple] = 1):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x


class Up(nn.Module):   
    def __init__(self, channels: int, filters: int, kernel_size: Union[int, tuple] = 3, 
                 padding: Union[int, tuple] = 1):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels, filters, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int]):
        super().__init__()
        self.num_layers = len(hidden_channels)

        self.down0 = Down(in_channels, hidden_channels[0])
        self.rnn0 = ConvLSTMCell(hidden_channels[0], hidden_channels[0])

        for i in range(1, self.num_layers):
            setattr(self, 'down' + str(i), Down(hidden_channels[i - 1], hidden_channels[i]))
            setattr(self, 'rnn' + str(i), ConvLSTMCell(hidden_channels[i], hidden_channels[i]))

    def forward(self, x: torch.Tensor) -> Tuple[list, list]:
        # x: 5D tensor (B, L, C, H, W)
        h_list = [None] * self.num_layers
        c_list = [None] * self.num_layers

        for i in range(x.size(1)):
            y = self.down0(x[:, i])
            h_list[0], c_list[0] = self.rnn0(y, h_list[0], c_list[0])
            for j in range(1, self.num_layers):
                y = getattr(self, 'down' + str(j))(h_list[j - 1])
                h_list[j], c_list[j] = getattr(self, 'rnn' + str(j))(y, h_list[j], c_list[j])
        
        return h_list, c_list


class Forecaster(nn.Module):
    def __init__(self, forecast_steps: int, out_channels: int, hidden_channels: List[int]):
        super().__init__()
        self.num_layers = len(hidden_channels)
        self.forecast_steps = forecast_steps

        for i in range(1, self.num_layers):
            setattr(self, 'rnn' + str(self.num_layers - i), ConvLSTMCell(hidden_channels[-i], hidden_channels[-i]))
            setattr(self, 'up' + str(self.num_layers - i), Up(hidden_channels[-i], hidden_channels[-i-1]))
        
        self.rnn0 = ConvLSTMCell(hidden_channels[0], hidden_channels[0])
        self.up0 = Up(hidden_channels[0], out_channels)
    
    def forward(self, h_list: list, c_list: list) -> torch.Tensor:
        output = []
        
        x = torch.zeros_like(h_list[-1], device=h_list[-1].device)
        for _ in range(self.forecast_steps):
            h_list[-1], c_list[-1] = getattr(self, 'rnn' + str(self.num_layers - 1))(x, h_list[-1], c_list[-1])
            y = getattr(self, 'up' + str(self.num_layers - 1))(h_list[-1])
            for j in range(1, self.num_layers):
                h_list[-j-1], c_list[-j-1] = getattr(self, 'rnn' + str(self.num_layers - j - 1))(y, h_list[-j-1], h_list[-j-1])
                y = getattr(self, 'up' + str(self.num_layers - j - 1))(h_list[-j-1])
            output.append(y)
    
        output = torch.stack(output, dim=0)
        output = torch.relu(output)
        # output: 5D tensor (B, L_out, C, H, W)
        output = output.transpose(1, 0)
        return output


class ConvLSTM(nn.Module):
    def __init__(self, forecast_steps: int, in_channels: int = 1, out_channels: int = 1, 
                 hidden_channels: List[int] = [64, 64, 64, 64]):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.forecaster = Forecaster(forecast_steps, out_channels, hidden_channels)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        states, cells = self.encoder(input_)
        output = self.forecaster(states, cells)
        return output
