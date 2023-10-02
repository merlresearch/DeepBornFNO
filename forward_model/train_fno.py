# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

sys.path.append("../")
import operator
from functools import partial, reduce
from glob import glob
from shutil import copyfile
from timeit import default_timer

import matplotlib.pyplot as plt
import torchvision
from Adam import Adam
from torch.utils.tensorboard import SummaryWriter

import dataloader
import utils

torch.manual_seed(0)
np.random.seed(0)
import os
import shutil

import configargparse
from model import *

from loss import LpLoss
from utils import *


def train(
    layer_num,
    batch_size,
    learning_rate,
    modes,
    width,
    type="born",
    root_path="./",
    dataset="v1",
    add_noise=0,
    step_size=700,
):
    """
    This function is the modified from Zongyi Li's code for work [paper](https://arxiv.org/pdf/2010.08895.pdf).
    """
    ################################################################
    # configs
    ################################################################

    ntrain = 350
    ntest = 50
    h = 63
    s = h
    num_freq = 50
    air_idx_val = 44

    epochs = 800
    gamma = 0.5

    summaries_dir = root_path
    writer = SummaryWriter(summaries_dir)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    print("using script", script_name)
    shutil.copyfile("{}.py".format(script_name), "{}/script.py".format(summaries_dir))

    checkpoints_dir = os.path.join(summaries_dir, "checkpoints")
    print("create checkpoint dir", checkpoints_dir)
    utils.cond_mkdir(checkpoints_dir)

    plan_index = 56
    ################################################################
    # load data and data normalization
    ################################################################
    if dataset == "v1":
        x, y, _, _ = dataloader.load_file_frequency(sample_num=(ntrain + ntest), type="scatter_field_f")
        x_train = torch.tensor(x[: ntrain * 50, :, :, :]).float()
        x_test = torch.tensor(x[ntrain * 50 :, :, :, :]).float()
        y_train = torch.tensor(y[: ntrain * 50, :, :, :]).float()
        y_test = torch.tensor(y[ntrain * 50 :, :, :, :]).float()

        x_normalizer = dataloader.UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = dataloader.UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )

    ################################################################
    # training and evaluation
    ################################################################
    if type == "born":
        model = FNO2d_weight_tight_born(modes, modes, width, layernum=layer_num).cuda()
    elif type == "vanilla":
        model = FNO2d(modes, modes, width, layernum=layer_num).cuda()
    else:
        print("model not available")
        raise NotImplementedError
    print(count_params(model))
    model.cuda()

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    total_steps = 0
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            if add_noise != 0:
                noise = torch.normal(mean=torch.zeros_like(x), std=add_noise)
                x_noise = x.clone()
                x_noise[:, :, :, 0] = x_normalizer.decode(x_noise)[:, :, :, 0]
                x_noise[:, :, :, 0] = x_noise[:, :, :, 0] + noise[:, :, :, 0]
                kz = np.random.randint(low=0, high=11) * 2 + 1
                blur = torchvision.transforms.GaussianBlur(kernel_size=(kz, kz)).cuda()
                x_noise[:, :, :, [0]] = blur(x_noise[:, :, :, [0]].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x_noise[:, air_idx_val:, :, [0]] = x_normalizer.decode(x)[:, air_idx_val:, :, [0]]
                x[:, :, :, 0] = x_normalizer.encode(x_noise)[:, :, :, 0]

            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            optimizer.zero_grad()
            out = model(x)
            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            total_steps += 1
            train_l2 += loss.item()
            writer.add_scalar("train_l2", loss.item(), total_steps)

        scheduler.step()

        if ep % 5 == 0:
            fig = plt.figure(figsize=[20, 3])
            for idx, bz in enumerate(range(0, batch_size, 3)[:8]):
                plt.subplot(1, 8, idx + 1)
                plt.plot(out[bz, plan_index, :, 0].detach().cpu().numpy())
                plt.plot(out[bz, plan_index, :, 1].detach().cpu().numpy())
                plt.plot(y[bz, plan_index, :, 0].detach().cpu().numpy(), "--", linewidth=3)
                plt.plot(y[bz, plan_index, :, 1].detach().cpu().numpy(), "--", linewidth=3)
            writer.add_figure("gt_vs_pred", fig, global_step=total_steps)

            grid = torchvision.utils.make_grid(x[:, :, :, 0].unsqueeze(1), normalize=True)
            writer.add_image("x", grid, global_step=total_steps)
            if add_noise != 0:
                grid = torchvision.utils.make_grid(noise[:, :, :, 0].unsqueeze(1), normalize=True)
                writer.add_image("noise", grid, global_step=total_steps)
            grid = torchvision.utils.make_grid(out[:, :, :, 0].unsqueeze(1), normalize=True)
            writer.add_image("predr", grid, global_step=total_steps)
            grid = torchvision.utils.make_grid(out[:, :, :, 1].unsqueeze(1), normalize=True)
            writer.add_image("predi", grid, global_step=total_steps)
            grid = torchvision.utils.make_grid(y[:, :, :, 0].unsqueeze(1), normalize=True)
            writer.add_image("gtr", grid, global_step=total_steps)
            grid = torchvision.utils.make_grid(y[:, :, :, 1].unsqueeze(1), normalize=True)
            writer.add_image("gti", grid, global_step=total_steps)

            model.eval()
            test_l2 = 0.0
            torch.save(
                model, "{}/fno_{}_{}_{}_{}_{}_current.pth".format(summaries_dir, type, layer_num, width, ntrain, ntest)
            )
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    batch_size = x.shape[0]
                    out = model(x)
                    test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

            train_l2 /= ntrain
            test_l2 /= ntest

            writer.add_scalar("average_test_l2", test_l2, total_steps)
            fig = plt.figure(figsize=[20, 3])
            for idx, bz in enumerate(range(0, batch_size, 4)[:8]):
                plt.subplot(1, 8, idx + 1)
                plt.plot(out[bz, plan_index, :, 0].detach().cpu().numpy())
                plt.plot(out[bz, plan_index, :, 1].detach().cpu().numpy())
                plt.plot(y[bz, plan_index, :, 0].detach().cpu().numpy(), "--", linewidth=3)
                plt.plot(y[bz, plan_index, :, 1].detach().cpu().numpy(), "--", linewidth=3)
            writer.add_figure("val_gt_vs_pred", fig, global_step=total_steps)

            grid = torchvision.utils.make_grid(out[:, :, :, 0].unsqueeze(1), normalize=True)
            writer.add_image("val_predr", grid, global_step=total_steps)
            grid = torchvision.utils.make_grid(out[:, :, :, 1].unsqueeze(1), normalize=True)
            writer.add_image("val_predi", grid, global_step=total_steps)
            grid = torchvision.utils.make_grid(y[:, :, :, 0].unsqueeze(1), normalize=True)
            writer.add_image("val_gtr", grid, global_step=total_steps)
            grid = torchvision.utils.make_grid(y[:, :, :, 1].unsqueeze(1), normalize=True)
            writer.add_image("val_gti", grid, global_step=total_steps)

            t2 = default_timer()
            print(ep, t2 - t1, train_l2, test_l2)

        if not ep % 50:
            torch.save(
                {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                os.path.join(checkpoints_dir, "model_epoch_%010d.pth" % ep),
            )
            print("storing model...........\n")


import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

p = configargparse.ArgumentParser()
p.add_argument("--layer_num", type=int, default=5, help="num of layer for FNO")
p.add_argument("--type", type=str, default="born", help="model type of FNO")
p.add_argument("--width", type=int, default=128, help="hidden feature size of FNO")
p.add_argument("--modes", type=int, default=12, help="number of frequencies of FNO")
p.add_argument("--dataset", type=str, default="v1", help="dataset")

p.add_argument("--lr", type=float, default=1 - 3, help="learning rate")
p.add_argument("--batch_size", type=int, default=64, help="batch size")
p.add_argument("--gpu", type=int, default=0, help="gpu id")
p.add_argument("--add_noise", type=float, default=0, help="gaussian noise added during training")
p.add_argument("--step_size", type=int, default=700, help="lr decay step size")

opt = p.parse_args()
print("--- Run Configuration ---")
torch.cuda.set_device(opt.gpu)

seed_everything(42)
fname_vars = ["dataset", "step_size", "add_noise", "type", "layer_num", "width", "modes", "batch_size", "lr"]
opt.experiment_name = "".join([f"{k}_{vars(opt)[k]}_|_".replace("[", "(").replace("]", ")") for k in fname_vars])[0:-1]
num_experiments = len(glob(os.path.join("./", opt.experiment_name) + "*"))
if num_experiments > 0:
    opt.experiment_name += f"_{num_experiments}"

# root_path = os.path.join("/media/data5/cyanzhao/gpr/forward_model/", opt.experiment_name)
root_path = os.path.join("../model_zoo/forward_model/", opt.experiment_name)
cond_mkdir(root_path)

################################################################
# Copy files for documentation
################################################################

utils.cond_mkdir("{}/files/".format(root_path))
copyfile("./train_fno.py", "{}/files/train_fno.py".format(root_path))
copyfile("./model.py", "{}/files/model.py".format(root_path))
p.write_config_file(opt, [os.path.join(root_path, "config.ini")])
print("finish copy files")

utils.seed_everything(42)

train(
    type=opt.type,
    layer_num=opt.layer_num,
    batch_size=opt.batch_size,
    learning_rate=opt.lr,
    width=opt.width,
    modes=opt.modes,
    root_path=root_path,
    dataset=opt.dataset,
    add_noise=opt.add_noise,
    step_size=opt.step_size,
)
