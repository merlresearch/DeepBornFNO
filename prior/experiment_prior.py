# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys

# Enable import from parent package
from email.policy import default

from encoder_model import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
from functools import partial
from glob import glob

import configargparse
import numpy as np
import torch
import train
from torch.utils.data import DataLoader

import dataloader
import loss
import utils

p = configargparse.ArgumentParser()
p.add("-c", "--config", required=False, is_config_file=True, help="Path to config file.")

p.add_argument("--logging_root", type=str, default="../model_zoo/priors/", help="root for logging")
p.add_argument(
    "--experiment_name",
    type=str,
    default="density_fno",
    help="Name of subdirectory in logging_root where summaries and checkpoints will be saved.",
)

# General model options
p.add_argument("--latent_size", type=int, default=64)
p.add_argument("--sig", type=float, default=0.001)
p.add_argument("--loss_mode", type=str, default="l2")
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--gpu", type=int, default=0, help="which gpu to use")
p.add_argument("--epochs_til_ckpt", type=int, default=10, help="Time interval in seconds until checkpoint is saved.")
p.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train for.")
p.add_argument(
    "--steps_til_summary", type=int, default=500, help="Time interval in seconds until tensorboard summary is saved."
)
p.add_argument("--lr", type=float, default=5e-4, help="learning rate. default=5e-4")

opt = p.parse_args()
# opt.use_pe = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# torch.cuda.set_device(opt.gpu)

np.random.seed(seed=121)

torch.manual_seed(121)
# torch.set_num_threads(1)

print("--- Run Configuration ---")
if opt.config is None:
    fname_vars = ["experiment_name", "batch_size", "sig", "latent_size", "lr", "loss_mode", "gpu"]
    opt.experiment_name = "".join([f"{k}_{vars(opt)[k]}_|_".replace("[", "(").replace("]", ")") for k in fname_vars])[
        16:-1
    ]

num_experiments = len(glob(os.path.join(opt.logging_root, opt.experiment_name) + "*"))
if num_experiments > 0:
    opt.experiment_name += f"_{num_experiments}"


root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

print("--- load dataset ---")
x, y, _, _ = dataloader.load_file_frequency(sample_num=400, type="scatter_field_f")
x_train = torch.tensor(x[: 350 * 50, :, :, :]).float()
x_test = torch.tensor(x[350 * 50 : 351 * 50, :, :, :]).float()
y_train = torch.tensor(y[: 350 * 50, :, :, :]).float()
y_test = torch.tensor(y[350 * 50 : 351 * 50, :, :, :]).float()

# here we adopt FNO normalization to make it compatible with FNO.
x_normalizer = dataloader.UnitGaussianNormalizer(x_train)

x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        x_train[:, :, :, [0]].permute(0, 3, 1, 2), x_train[:, :, :, [0]].permute(0, 3, 1, 2)
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test[:, :, :, [0]].permute(0, 3, 1, 2), x_test[:, :, :, [0]].permute(0, 3, 1, 2)),
    batch_size=opt.batch_size,
    shuffle=False,
)

model = ConvAutoencoder(1, opt.latent_size).cuda()
loss_fn = partial(loss.encoder, opt.loss_mode, opt.sig)

summary_fn = partial(utils.summary_fn_cnn_encoder, x_normalizer, True, opt.latent_size)

# Save command-line parameters log directory.
p.write_config_file(opt, [os.path.join(root_path, "config.ini")])
with open(os.path.join(root_path, "model.txt"), "w") as out_file:
    out_file.write(str(model))

train.train(
    model=model,
    train_dataloader=train_loader,
    epochs=opt.num_epochs,
    lr=opt.lr,
    steps_til_summary=opt.steps_til_summary,
    epochs_til_checkpoint=opt.epochs_til_ckpt,
    model_dir=root_path,
    loss_fn=loss_fn,
    summary_fn=summary_fn,
)
