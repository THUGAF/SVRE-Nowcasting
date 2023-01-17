import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SpatioTemporalLSTMCell, MotionGRU
from .layers import reshape_patch, reshape_patch_back


# refers to https://github.com/thuml/MotionRNN
class MotionRNN(nn.Module):
    def __init__(self, forecast_steps, img_height, img_width, num_layers=4, num_hidden=64, patch_size=1):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.patch_size = patch_size
        self.patch_height = img_height // patch_size
        self.patch_width = img_width // patch_size
        self.patch_ch = patch_size ** 2
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.neighbour = 3
        self.motion_hidden = 2 * self.neighbour * self.neighbour

        cell_list = []
        for i in range(num_layers):
            in_channel = self.patch_ch if i == 0 else num_hidden
            cell_list.append(SpatioTemporalLSTMCell(in_channel, num_hidden))
        enc_list = []
        for i in range(num_layers - 1):
            enc_list.append(nn.Conv2d(num_hidden, num_hidden // 4, kernel_size=3, stride=2, padding=1))
        motion_list = []
        for i in range(num_layers - 1):
            motion_list.append(MotionGRU(num_hidden // 4, self.motion_hidden, self.neighbour))
        dec_list = []
        for i in range(num_layers - 1):
            dec_list.append(nn.ConvTranspose2d(num_hidden // 4, num_hidden, kernel_size=4, stride=2, padding=1))
        gate_list = []
        for i in range(num_layers - 1):
            gate_list.append(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=3, stride=1, padding=1))
        
        self.gate_list = nn.ModuleList(gate_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.motion_list = nn.ModuleList(motion_list)
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)
        self.conv_first_v = nn.Conv2d(self.patch_ch, num_hidden, 1, stride=1, padding=0, bias=False)
        self.conv_last = nn.Conv2d(num_hidden, self.patch_ch, 1, stride=1, padding=0, bias=False)


    def forward(self, x):
        x = reshape_patch(x, self.patch_size)
        batch_size, input_steps = x.size(0), x.size(1)
        output = []
        h_t = []
        c_t = []
        h_t_conv = []
        h_t_conv_offset = []
        mean = []

        for i in range(self.num_layers):
            zeros = torch.empty([batch_size, self.num_hidden, self.patch_height, self.patch_width]).to(x.device)
            nn.init.xavier_normal_(zeros)
            h_t.append(zeros)
            c_t.append(zeros)

        for i in range(self.num_layers - 1):
            zeros = torch.empty([batch_size, self.num_hidden // 4, self.patch_height // 2, self.patch_width // 2]).to(x.device)
            nn.init.xavier_normal_(zeros)
            h_t_conv.append(zeros)
            zeros = torch.empty([batch_size, self.motion_hidden, self.patch_height // 2, self.patch_width // 2]).to(x.device)
            nn.init.xavier_normal_(zeros)
            h_t_conv_offset.append(zeros)
            mean.append(zeros)

        mem = torch.empty([batch_size, self.num_hidden, self.patch_height, self.patch_width]).to(x.device)
        motion_highway = torch.empty([batch_size, self.num_hidden, self.patch_height, self.patch_width]).to(x.device)
        nn.init.xavier_normal_(mem)
        nn.init.xavier_normal_(motion_highway)

        for t in range(input_steps):
            net = x[:, t]
            motion_highway = self.conv_first_v(net)
            h_t[0], c_t[0], mem, motion_highway = self.cell_list[0](net, h_t[0], c_t[0], mem, motion_highway)
            net = self.enc_list[0](h_t[0])
            h_t_conv[0], h_t_conv_offset[0], mean[0] = self.motion_list[0](net, h_t_conv_offset[0], mean[0])
            h_t_tmp = self.dec_list[0](h_t_conv[0])
            o_t = torch.sigmoid(self.gate_list[0](torch.cat([h_t_tmp, h_t[0]], dim=1)))
            h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

            for i in range(1, self.num_layers - 1):
                h_t[i], c_t[i], mem, motion_highway = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], mem, motion_highway)
                net = self.enc_list[i](h_t[i])
                h_t_conv[i], h_t_conv_offset[i], mean[i] = self.motion_list[i](net, h_t_conv_offset[i], mean[i])
                h_t_tmp = self.dec_list[i](h_t_conv[i])
                o_t = torch.sigmoid(self.gate_list[i](torch.cat([h_t_tmp, h_t[i]], dim=1)))
                h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]

            h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway = self.cell_list[self.num_layers - 1](
                h_t[self.num_layers - 2], h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
        
        output.append(x_gen)
        for t in range(self.forecast_steps - 1):
            net = x_gen
            motion_highway = self.conv_first_v(net)
            h_t[0], c_t[0], mem, motion_highway = self.cell_list[0](net, h_t[0], c_t[0], mem, motion_highway)
            net = self.enc_list[0](h_t[0])
            h_t_conv[0], h_t_conv_offset[0], mean[0] = self.motion_list[0](net, h_t_conv_offset[0], mean[0])
            h_t_tmp = self.dec_list[0](h_t_conv[0])
            o_t = torch.sigmoid(self.gate_list[0](torch.cat([h_t_tmp, h_t[0]], dim=1)))
            h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

            for i in range(1, self.num_layers - 1):
                h_t[i], c_t[i], mem, motion_highway = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], mem, motion_highway)
                net = self.enc_list[i](h_t[i])
                h_t_conv[i], h_t_conv_offset[i], mean[i] = self.motion_list[i](net, h_t_conv_offset[i], mean[i])
                h_t_tmp = self.dec_list[i](h_t_conv[i])
                o_t = torch.sigmoid(self.gate_list[i](torch.cat([h_t_tmp, h_t[i]], dim=1)))
                h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]

            h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway = self.cell_list[self.num_layers - 1](
                h_t[self.num_layers - 2], h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            output.append(x_gen)

        output = torch.stack(output, dim=0).transpose(1, 0)
        output = reshape_patch_back(output, self.patch_size)
        return output
