# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import enum
import operator
import os
from functools import reduce
from random import sample
from types import new_class

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def plt_colorbar(matrix, vmax=None, vmin=None, cmap="viridis"):
    if vmax != None:
        im = plt.imshow(matrix, vmax=vmax, vmin=vmin, cmap=cmap)
    else:
        im = plt.imshow(matrix, cmap=cmap)
    im_ratio = matrix.shape[0] / matrix.shape[1]
    plt.colorbar(im, fraction=0.046 * im_ratio, pad=0.04)


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


def summary_fn_cnn_encoder(
    normalizer,
    sample,
    latent_dim,
    model,
    model_input,
    gt,
    model_output,
    writer,
    total_steps,
    prefix="train_",
):
    with torch.no_grad():
        model_output = model(model_input)
        model_output = model_output["output"]
    grid = torchvision.utils.make_grid(model_output[:, 0, :, :].unsqueeze(1), scale_each=False, normalize=True)
    writer.add_image("{}_pred".format(prefix), grid, global_step=total_steps)

    grid = torchvision.utils.make_grid(gt[:, 0, :, :].unsqueeze(1), scale_each=False, normalize=True)
    writer.add_image("{}_gt".format(prefix), grid, global_step=total_steps)

    if normalizer != None:
        model_output = normalizer.decode(model_output.cpu().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        gt = normalizer.decode(gt.cpu().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        grid = torchvision.utils.make_grid(model_output[:, 0, :, :].unsqueeze(1), scale_each=False, normalize=True)
        writer.add_image("{}_pred_norm".format(prefix), grid, global_step=total_steps)

        grid = torchvision.utils.make_grid(gt[:, 0, :, :].unsqueeze(1), scale_each=False, normalize=True)
        writer.add_image("{}_gt_norm".format(prefix), grid, global_step=total_steps)

    if sample and prefix == "val":
        latent = torch.randn(model_output.shape[0], latent_dim).cuda()
        with torch.no_grad():
            model_output = model.decode(latent)
        grid = torchvision.utils.make_grid(model_output[:, 0, :, :].unsqueeze(1), scale_each=False, normalize=True)
        writer.add_image("{}_sampled".format(prefix), grid, global_step=total_steps)
        if normalizer != None:
            model_output = normalizer.decode(model_output.cpu().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            grid = torchvision.utils.make_grid(model_output[:, 0, :, :].unsqueeze(1), scale_each=False, normalize=True)
            writer.add_image("{}_sampled_norm".format(prefix), grid, global_step=total_steps)
