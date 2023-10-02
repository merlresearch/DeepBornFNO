# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import os
from builtins import complex
from curses import noraw

import numpy as np
import torch
from genericpath import samefile
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py


# normalization by FNO, pointwise gaussian
class UnitGaussianNormalizer(object):
    """
    @author: Zongyi Li
    This function is the taken from Zongyi Li's code for work [paper](https://arxiv.org/pdf/2010.08895.pdf).
    """

    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


def load_file_frequency(
    file_path="../data/v1/data_frequency.h5",
    file_path_free="../data/v1/data_frequency_freespace.h5",
    sample_num=10,
    start_num=0,
    type="total_field_f",
    normalize=1,
):

    import os

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    import h5py

    if normalize == 1:
        complex_coeff = np.load("../data/v1/normalize_complex_coeff.npy")
    else:
        complex_coeff = 1

    with h5py.File(file_path_free, "r") as hff:
        free_field = hff["f_field"][:]
        free_field_freq = hff["frequency"][:]

    with h5py.File(file_path, "r") as hf:
        num_sample = len(hf.keys())
        frequency_steps = 50
        channel = 3
        store_data_input = np.zeros([sample_num, frequency_steps, 63, 63, channel])
        store_data_output = np.zeros([sample_num, frequency_steps, 63, 63, 2])
        free_field = free_field / complex_coeff
        for i, keys in enumerate(range(start_num, start_num + sample_num)):
            keys = str(keys)
            eps = hf[keys]["eps"][:]
            f_field = hf[keys]["f_field"][:]
            store_data_input[i, :, :, :, 0] = np.repeat(eps[None, :, :], 50, axis=0)
            store_data_input[i, :, :, :, 1] = free_field.real
            store_data_input[i, :, :, :, 2] = free_field.imag
            if type == "total_field_f":
                f_field = f_field / complex_coeff
                store_data_output[i, :, :, :, 0] = (f_field).real
                store_data_output[i, :, :, :, 1] = (f_field).imag
            elif type == "scatter_field_f":
                f_field = f_field / complex_coeff
                store_data_output[i, :, :, :, 0] = (f_field - free_field).real
                store_data_output[i, :, :, :, 1] = (f_field - free_field).imag
    store_data_input = store_data_input.reshape([-1, 63, 63, 3])
    store_data_output = store_data_output.reshape([-1, 63, 63, 2])
    return store_data_input, store_data_output, free_field, free_field_freq
