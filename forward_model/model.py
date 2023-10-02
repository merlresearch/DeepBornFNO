# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2023 NeuralOperator developers

# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    """
    This function is the taken from Zongyi Li's code for work [paper](https://arxiv.org/pdf/2010.08895.pdf).
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    This function is the modified from Zongyi Li's code for work [paper](https://arxiv.org/pdf/2010.08895.pdf).
    """

    def __init__(self, modes1, modes2, width, layernum=5):
        super(FNO2d, self).__init__()

        """
        The overall network. It can $layernum of layers of the Fourier layer.
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(5, self.width)  # input channel is 3: (a(x, y), x, y)

        self.convs = []
        self.ws = []
        self.layernum = layernum
        for i in range(self.layernum):
            self.convs.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.ws.append(nn.Conv2d(self.width, self.width, 1))
        self.convs = nn.Sequential(*self.convs)
        self.ws = nn.Sequential(*self.ws)

        self.fc1 = nn.Linear(self.width, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for idx in range(self.layernum - 1):
            x1 = self.convs[idx](x)
            x2 = self.ws[idx](x)
            x = x1 + x2
            x = F.gelu(x)

        x1 = self.convs[-1](x)
        x2 = self.ws[-1](x)
        x = x1 + x2

        x = x[..., : -self.padding, : -self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class SpectralConv2d_born(nn.Module):
    """
    This function is the modified from Zongyi Li's code for work [paper](https://arxiv.org/pdf/2010.08895.pdf).
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_born, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, x_eps):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x * x_eps)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d_weight_tight_born(nn.Module):
    """
    This function is the modified from Zongyi Li's code for work [paper](https://arxiv.org/pdf/2010.08895.pdf).
    """

    def __init__(self, modes1, modes2, width, layernum=4):
        super(FNO2d_weight_tight_born, self).__init__()

        """
        The overall network. It contains $layer layers of the Born Fourier layer (share the same weight).
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc_inc = nn.Linear(4, self.width)  # input channel is 3: (a(x, y), x, y)
        self.fc_eps = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d_born(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.layernum = layernum

        self.fc1 = nn.Linear(self.width, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x_in = torch.cat((x[:, :, :, 1:], grid), dim=-1)
        x_eps = torch.cat((x[:, :, :, [0]], grid), dim=-1)
        x_in = self.fc_inc(x_in)
        x_eps = self.fc_eps(x_eps)
        x_in = x_in.permute(0, 3, 1, 2)
        x_in = F.pad(x_in, [0, self.padding, 0, self.padding])
        x_eps = x_eps.permute(0, 3, 1, 2)
        x_eps = F.pad(x_eps, [0, self.padding, 0, self.padding])

        incident = x_in.clone()
        x = x_in
        for _ in range(self.layernum - 1):
            x = self.conv0(x, x_eps)
            x = self.w1(F.leaky_relu(self.w0(x)))
            x = F.leaky_relu(x)
            x = x + incident

        x = self.conv0(x, x_eps)
        x = self.w1(F.leaky_relu(self.w0(x)))
        x = F.leaky_relu(x)
        x_in = x + incident

        x_in = x_in[..., : -self.padding, : -self.padding]
        x_in = x_in.permute(0, 2, 3, 1)
        x_in = self.fc1(x_in)
        x_in = F.leaky_relu(x_in)
        x_in = self.fc2(x_in)
        return x_in

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
