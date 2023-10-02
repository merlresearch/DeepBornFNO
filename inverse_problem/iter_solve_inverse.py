# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import pdb
import sys
from logging import root
from pickle import FALSE
from statistics import mean
from unittest import load_tests

import numpy as np
import torch

sys.path.append("../")
from utils import *

sys.path.append("../forward_model/")
import matplotlib.pyplot as plt
from model import *

import dataloader

sys.path.append("")
from functools import partial
from glob import glob
from os.path import exists

import matplotlib.animation as animation
from solve_inverse import *

import loss

sys.path.append("../prior/")
import shutil

import configargparse
from encoder_model import *


def animate_iter(pars, normalizer, path):
    fig = plt.figure(figsize=(8, 8))
    par = normalizer.decode(pars[0])
    im = plt.imshow(par[0, :, :, 0].squeeze(), interpolation="none", aspect="auto", vmin=1, vmax=10)

    def animate_func(i):
        par = normalizer.decode(pars[i])
        im.set_array(par[0, :, :, 0])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=len(pars) - 1,
        interval=100,  # in ms
    )
    writergif = animation.PillowWriter(fps=2)
    anim.save("{}/animate.gif".format(path), writer=writergif)


def solve_all_inverse_problem(
    prior,
    optimizer,
    lr,
    num_iter,
    pivit,
    pivit_iter,
    reg_coeff,
    mask,
    data_type,
    root_path=None,
    incremental_freq=False,
    test=False,
    sweep_par=None,
    drop_reg_high=False,
    loader_type="test",
    noise=0,
    type="vanilla",
    filter=False,
    gamma=1,
    latent=64,
    data_path=None,
    noise_src=0,
    y_norm=None,
    fno_par=[128, 15, 0.0],
    fno_noise=0.0,
    load_weight=True,
):

    print("--load dataset--")
    batch_size = 50
    x, y, _, _ = dataloader.load_file_frequency(sample_num=400, type=data_type)
    x_train = torch.tensor(x[: 350 * 50, :, :, :]).float()
    x_test = torch.tensor(x[350 * 50 :, :, :, :]).float()
    y_train = torch.tensor(y[: 350 * 50, :, :, :]).float()
    y_test = torch.tensor(y[350 * 50 :, :, :, :]).float()

    x_normalizer = dataloader.UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = dataloader.UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )

    x_original, y_original, _, _ = dataloader.load_file_frequency(
        start_num=350, sample_num=50, type=data_type, normalize=0
    )
    y_test = torch.tensor(y_original).float()
    y_test = y_normalizer.encode(y_test)
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )

    print("--load loss fn --")
    loss_fn = partial(loss.mse)

    if type == "born":
        print("load born")
        log_dir = glob(
            "../model_zoo/forward_model/*noise_{}*born*/fno_born_{}_{}_350_50_current.pth".format(
                fno_noise, fno_par[0], fno_par[1]
            )
        )[0]
        model = (torch.load(log_dir, map_location="cuda:0")).cuda()
        model.eval()
    elif type == "vanilla":
        print("load vanilla")
        log_dir = glob(
            "../model_zoo/forward_model/*noise_{}*vanilla*/fno_vanilla_{}_{}_350_50_current.pth".format(
                fno_noise, fno_par[0], fno_par[1]
            )
        )[0]
        model = (torch.load(log_dir, map_location="cuda:0")).cuda()
        model.eval()

    if prior == "autoencoder":
        print("--load pre-train autoencoder--")
        latent_size = latent
        prior_path = glob("../model_zoo/priors/*latent_size_{}*/*/model_final.pth".format(latent_size))[0]
        prior = ConvAutoencoder(1, latent_size).cuda()
        prior.load_state_dict(torch.load(prior_path))
        prior.eval()
    else:
        prior = None

    complex_weight = np.load("../data/v1/normalize_complex_coeff.npy")
    weight = torch.zeros([50, 1, 1, 2]).cuda()
    weight[:, :, :, 0] = torch.tensor(complex_weight.real).cuda()
    weight[:, :, :, 1] = torch.tensor(complex_weight.imag).cuda()

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    shutil.copyfile("{}.py".format(script_name), "{}/iter_solve_inverse.py".format(root_path))
    shutil.copyfile("solve_inverse.py", "{}/solve_inverse.py".format(root_path))

    y_norm = y_normalizer
    y_norm.cuda()

    print("--generate mask---")
    if mask == "all":
        mask = torch.ones([63, 63, 2]).cuda()
    elif mask == "line":
        mask = torch.zeros([63, 63, 2]).cuda()
        mask[56, :, :] = 1
    elif mask == "dash":
        mask = torch.zeros([63, 63, 2]).cuda()
        mask[56, ::5, :] = 1
    elif mask == "dash5":
        mask = torch.zeros([63, 63, 2]).cuda()
        mask[56, 1::5, :] = 1
    elif mask == "dash10":
        mask = torch.zeros([63, 63, 2]).cuda()
        mask[56, ::10, :] = 1
    elif mask == "d10_2":
        mask = torch.zeros([63, 63, 2]).cuda()
        mask[56, 5:-5:10, :] = 1
    elif mask == "mono":
        mask = torch.zeros([63, 63, 2]).cuda()
        mask[56, 32, :] = 1

    plt.imshow(mask[:, :, 0].detach().cpu().numpy())
    plt.colorbar()
    plt.savefig("{}/mask.png".format(root_path))
    loader = test_loader
    print(len(loader))
    for i, (x, y) in enumerate(loader):
        root_path_i = "{}/{}".format(root_path, i)
        print("check if exist", exists("{}/eps_estimated.npy".format(root_path_i)))
        if exists("{}/eps_estimated.npy".format(root_path_i)):
            print("exist!!!!!,{}".format(i))
            continue
        utils.cond_mkdir(root_path_i)

        gt_dynamics = y

        inputs = x[:, :, :, 1:]

        plt.figure(figsize=[20, 5])
        plt.subplot(141)
        plt.imshow((gt_dynamics[:, 56, :, 0].cpu().numpy()))
        plt.title("real")
        plt.subplot(142)
        plt.imshow((gt_dynamics[:, 56, :, 1].cpu().numpy()))
        plt.title("imag")
        plt.subplot(143)
        plt.plot((gt_dynamics[:, 56, 56, 0].cpu().numpy()))
        plt.title("real")
        plt.plot((gt_dynamics[:, 56, 56, 1].cpu().numpy()), "--")
        plt.title("imag")
        plt.subplot(144)
        plt.plot((gt_dynamics[:, 56, 20, 0].cpu().numpy()))
        plt.title("real")
        plt.plot((gt_dynamics[:, 56, 20, 1].cpu().numpy()), "--")
        plt.title("imag")
        plt.savefig("{}/observation.png".format(root_path_i))

        gt_eps = x[0, :, :, [0]].cuda()
        print("!!!!!!!!!!!!!begin itering!!!!!!!!!!!!!")
        eps_estimated, losses, mse, mses_decoded = solve_inverse_frequency(
            model,
            inputs,
            gt_dynamics,
            loss_fn,
            mask=mask,
            num_iter=num_iter,
            prior=prior,
            init_val=1,
            device="cuda",
            lr=lr,
            plot=True,
            gt_eps=gt_eps,
            normalizer=x_normalizer,
            pivit=pivit,
            optimizer=optimizer,
            pivit_iter=pivit_iter,
            incremental_freq=incremental_freq,
            save=True,
            root_path=root_path_i,
            tv_reg=reg_coeff,
            drop_reg_high=drop_reg_high,
            filter=filter,
            gamma=gamma,
            y_normalizer=y_norm,
            weight=weight,
            load_weight=load_weight,
        )
        plt.figure()
        plt.subplot(121)
        plt.plot(losses)
        plt.title("obj: {}".format(losses[-1]))
        plt.yscale("log")
        plt.subplot(122)
        plt.plot(mse)
        plt.title("mse: {}".format(mse[-1]))
        plt.yscale("log")
        plt.savefig("{}/loss_obj.png".format(root_path_i))
        np.save("{}/eps_estimated.npy".format(root_path_i), eps_estimated[-1].detach().cpu().numpy())
        f = open("{}/summary.txt".format(root_path), "a+")
        content = str([i, losses[-1], mse[-1]])
        f.write(content)
        f.write(" \n")
        f.close()

        f = open("{}/summary_decoded.txt".format(root_path), "a+")
        content = str([i, losses[-1], mses_decoded[-1]])
        f.write(content)
        f.write(" \n")
        f.close()

        if sweep_par != None:
            f = open("log/summary_{}.txt".format(sweep_par), "w+")
            content = str(
                [i, losses[-1], mse[-1], lr, optimizer, num_iter, pivit, pivit_iter, reg_coeff, drop_reg_high]
            )
            f.write(content)
            f.write(" \n")
            f.close()
        plt.close("all")
        animate_iter(eps_estimated, x_normalizer, root_path_i)
        plt.close("all")
        if test:
            break
        if i == 50:
            break

    return


p = configargparse.ArgumentParser()
p.add("-c", "--config", required=False, is_config_file=True, help="Path to config file.")
p.add_argument("--prior", type=str, default=None, help="piror we use")
p.add_argument("--optimizer", type=str, default="adam", help="optimizer we use")
p.add_argument("--lr", type=float, default=1e-2, help="lr rate")
p.add_argument("--num_iter", type=int, default=500, help="num of iteration")
p.add_argument("--pivit", type=int, default=0, help="if use pivit")
p.add_argument("--pivit_iter", type=int, default=200, help="iteration when begin pivit")
p.add_argument("--reg_coeff", type=float, default=0.001, help="coefficient for TV reg")
p.add_argument("--mask", type=str, default="line_dot", help="mask")
p.add_argument("--data_type", type=str, default="scatter_field_f", help="mask")
p.add_argument("--in_f", type=int, default=0, help="incremental_freq")
p.add_argument("--test", type=int, default=0, help="if test")
p.add_argument("--sweep_par", type=str, default=None, help="mask")
p.add_argument("--drop_reg_high", type=int, default=0, help="if drop_reg_high")
p.add_argument("--loader_type", type=str, default="test", help="mask")
p.add_argument("--type", type=str, default="vanilla", help="mask")
p.add_argument("--gpu", type=int, default=0, help="mask")
p.add_argument("--note", type=str, default="note", help="mask")
p.add_argument("--filter", type=int, default=0, help="mask")
p.add_argument("--gamma", type=float, default=1, help="mask")
p.add_argument("--latent", type=int, default=64, help="mask")
p.add_argument("--data_path", type=str, default="None", help="data_path")
p.add_argument("--noise_src", type=float, default=0, help="data_path")
p.add_argument("--y_norm", type=int, default=0, help="data_path")
p.add_argument("--fno_par", type=int, nargs=2, default=[15, 128], help="data_path")
p.add_argument("--fno_noise", type=float, default=0.0, help="noise in fno training")
p.add_argument("--norm", type=int, default=2, help="data_path")
p.add_argument("--load_weight", type=int, default=1, help="data_path")

opt = p.parse_args()
print("--- Run Configuration ---")

torch.cuda.set_device(opt.gpu)
fname_vars = [
    "type",
    "optimizer",
    "norm",
    "fno_par",
    "fno_noise",
    "latent",
    "gamma",
    "filter",
    "noise_src",
    "prior",
    "lr",
    "num_iter",
    "pivit",
    "pivit_iter",
    "reg_coeff",
    "mask",
    "in_f",
]
opt.experiment_name = "".join([f"{k}_{vars(opt)[k]}_|_".replace("[", "(").replace("]", ")") for k in fname_vars])[0:-1]

num_experiments = len(glob(os.path.join("./log/log_{}_{}/".format(opt.note, opt.type), opt.experiment_name) + "*"))

root_path = os.path.join("./log/log_{}_{}/".format(opt.note, opt.type), opt.experiment_name)
utils.cond_mkdir(root_path)
utils.seed_everything(42)

p.write_config_file(opt, [os.path.join(root_path, "config.ini")])
torch.set_num_threads(2)

solve_all_inverse_problem(
    prior=opt.prior,
    optimizer=opt.optimizer,
    lr=opt.lr,
    num_iter=opt.num_iter,
    pivit=opt.pivit == 1,
    pivit_iter=opt.pivit_iter,
    reg_coeff=opt.reg_coeff,
    mask=opt.mask,
    data_type=opt.data_type,
    root_path=root_path,
    incremental_freq=opt.in_f,
    test=opt.test == 1,
    sweep_par=opt.sweep_par,
    drop_reg_high=opt.drop_reg_high == 1,
    loader_type=opt.loader_type,
    filter=opt.filter == 1,
    gamma=opt.gamma,
    latent=opt.latent,
    data_path=opt.data_path,
    noise_src=opt.noise_src,
    y_norm=opt.y_norm == 1,
    fno_par=opt.fno_par,
    fno_noise=opt.fno_noise,
    load_weight=opt.load_weight == 1,
    type=opt.type,
)
