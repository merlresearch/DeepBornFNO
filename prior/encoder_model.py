# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2020 Anand Krishnamoorthy Subramanian

# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0

import sys

sys.path.append("../")
import torch
from torch import nn
from torch.nn import functional as F

"""
The function is modified from
https://github.com/AntixK/PyTorch-VAE/blob/8bf7be5c83c35a2027aa1771e6f0fb1ad9425536/models/vanilla_vae.py
"""


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=None, resolution=63):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(inplace=False),
                )
            )

            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_latent = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(inplace=False),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels, kernel_size=3, padding=1),
        )
        self.resolution = resolution

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        latent = self.fc_latent(torch.flatten(result, start_dim=1))
        return latent

    def decode(self, z):
        latent = z.clone()
        result = self.decoder_input(latent)
        result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        result_bf = result.clone()
        result = self.final_layer(result_bf)
        result = result[:, :, : self.resolution, : self.resolution]
        return result

    def forward(self, input):
        z = self.encode(input)
        return {"output": self.decode(z), "latent": z}
