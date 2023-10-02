# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
from logging import root
from turtle import numinput
from xml.sax.xmlreader import InputSource

import numpy as np
import torch
from tqdm.autonotebook import tqdm

sys.path.append("../")
import matplotlib.pyplot as plt
import torchvision

import utils


def solve_inverse_frequency(
    model,
    inputs,
    gt_dynamics,
    loss_fn,
    mask=1,
    num_iter=1000,
    prior=None,
    init_val=1,
    device="cuda",
    lr=1e-4,
    plot=True,
    total=False,
    gt_eps=None,
    normalizer=None,
    pivit=False,
    optimizer="adam",
    tv_reg=0,
    incremental_freq=False,
    pivit_iter=200,
    save=False,
    root_path=None,
    drop_reg_high=False,
    filter=False,
    gamma=1,
    y_normalizer=None,
    weight=None,
    load_weight=False,
):
    """
    model: the forward model, takes 'inputs' as input and predict the frequency fields
    inputs: the input of the model
            for the fno: num_freq*N*N*2
    gt_dynmaics: the ground truth frequency field at each frequency: num_freq*N*N*2
    mask: measurements mask: num_freq*N*N*2
    num_iter: number of iteration for optimization
    prior: if we have prior for the eps
    """

    num_f, nx, nz, _ = gt_dynamics.shape
    if incremental_freq == 1:
        num_f_list = np.linspace(10, num_f, 5)
        repeat = int(num_iter / 5)
        num_f_list = np.repeat(num_f_list, repeat)
    elif incremental_freq == 0:
        num_f_list = np.ones(num_iter) * num_f

    if prior == None:

        class estimated_par(torch.nn.Module):
            def __init__(self):
                super(estimated_par, self).__init__()
                self.estimated_par = torch.nn.Parameter(
                    (torch.rand([1, nx, nz, 1])).to(device) * init_val, requires_grad=True
                ).to(device)

    else:

        class estimated_par(torch.nn.Module):
            def __init__(self):
                super(estimated_par, self).__init__()
                self.estimated_par = torch.nn.Parameter(
                    (torch.zeros([1, prior.latent_dim]).to(device)), requires_grad=True
                ).to(device)

    observation = mask * gt_dynamics.to(device)
    par = estimated_par().to(device)
    if optimizer == "adam":
        optim = torch.optim.Adam(params=par.parameters(), lr=lr)
    elif optimizer == "sgd":
        optim = torch.optim.SGD(params=par.parameters(), lr=lr)

    schedular = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=gamma, last_epoch=-1, verbose=True)
    loss = []
    mses = []
    mses_decoded = []
    estimate_pars = []
    with tqdm(total=num_iter) as pbar:

        for total_steps in range(num_iter):
            num_f_t = num_f_list[min(num_iter - 1, total_steps)]
            num_f_t = int(num_f_t)
            if num_f_t == 50 and drop_reg_high:
                tv_reg = 0
            if prior == None:
                estimate_par = par.estimated_par.repeat([num_f_t, 1, 1, 1])
                currnet_input = torch.cat([estimate_par, inputs[:num_f_t, :, :, :].to(device)], dim=-1)
            else:
                estimate_par = prior.decode(par.estimated_par)
                if filter:
                    if num_f_t <= 10:
                        blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 9)).cuda()
                        estimate_par = blur(estimate_par)
                    elif num_f <= 30:
                        blur = torchvision.transforms.GaussianBlur(kernel_size=(3, 6)).cuda()
                        estimate_par = blur(estimate_par)
                    elif num_f <= 40:
                        blur = torchvision.transforms.GaussianBlur(kernel_size=(2, 3)).cuda()
                        estimate_par = blur(estimate_par)
                estimate_par = estimate_par.permute(0, 2, 3, 1)
                currnet_input = torch.cat(
                    [estimate_par.repeat([num_f_t, 1, 1, 1]), inputs[:num_f_t, :, :, :].to(device)], dim=-1
                )

            estimated_dynamic = model(currnet_input).reshape(num_f_t, 63, 63, 2)
            if load_weight:
                re_norm_estimated_dynamic = y_normalizer.decode(estimated_dynamic)
                rescale_y = torch.zeros_like(re_norm_estimated_dynamic)
                rescale_y[:, :, :, 0] = (
                    re_norm_estimated_dynamic[:, :, :, 0] * weight[:num_f_t, :, :, 0]
                    - re_norm_estimated_dynamic[:, :, :, 1] * weight[:num_f_t, :, :, 1]
                )
                rescale_y[:, :, :, 1] = (
                    re_norm_estimated_dynamic[:, :, :, 0] * weight[:num_f_t, :, :, 1]
                    + re_norm_estimated_dynamic[:, :, :, 1] * weight[:num_f_t, :, :, 0]
                )
                estimated_dynamic = y_normalizer.encode(rescale_y + 1e-16).clone()

            # if y_normalizer!=None:
            #     estimated_dynamic = y_normalizer.decode(estimated_dynamic)
            losses = loss_fn(
                estimated_dynamic, observation[:num_f_t, :, :, :], mask, estimate_par=estimate_par, tv_reg=tv_reg
            )
            loss_t = 0
            for key in losses.keys():
                loss_t += losses[key]
            optim.zero_grad()
            loss_t.backward()

            loss.append(loss_t.detach().cpu().numpy())
            if prior == None:
                estimate_par = par.estimated_par
            else:
                estimate_par = prior.decode(par.estimated_par)
                if filter:
                    if num_f_t <= 10:
                        blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 9)).cuda()
                        estimate_par = blur(estimate_par)
                    elif num_f <= 30:
                        blur = torchvision.transforms.GaussianBlur(kernel_size=(3, 6)).cuda()
                        estimate_par = blur(estimate_par)
                    elif num_f <= 40:
                        blur = torchvision.transforms.GaussianBlur(kernel_size=(2, 3)).cuda()
                        estimate_par = blur(estimate_par)
                estimate_par = estimate_par.permute(0, 2, 3, 1)
            mses.append(((estimate_par - gt_eps) ** 2).mean().detach().cpu().numpy())
            gt_eps_denorm = normalizer.decode(gt_eps.cpu())
            estimate_par_denorm = normalizer.decode(estimate_par.cpu())
            mses_decoded.append(((gt_eps_denorm[:, :, 0] - estimate_par_denorm[:, :, :, 0]) ** 2).mean())
            torch.nn.utils.clip_grad_norm_(par.parameters(), 0.01)
            optim.step()

            if total_steps % 500 == 0:
                schedular.step()
            pbar.update(1)

            if total_steps % 50 == 0:
                estimate_pars.append(estimate_par.detach().cpu())
            if total_steps % 400 == 0 and plot:
                print("use {} frequenies".format(num_f_t))
                plt.figure()
                plt.subplot(131)
                gt_eps_denorm = normalizer.decode(gt_eps.cpu())
                utils.plt_colorbar(gt_eps_denorm[:, :, 0].squeeze().detach().cpu().numpy(), vmin=1, vmax=10)
                plt.title("gt,use{}".format(num_f_t))
                plt.subplot(132)
                estimate_par_denorm = normalizer.decode(estimate_par.cpu())
                utils.plt_colorbar(estimate_par_denorm[:, :, :, 0].squeeze().detach().cpu().numpy(), vmin=1, vmax=10)
                plt.title("iter:{},gt".format(total_steps))

                if gt_eps != None:
                    mse = loss_fn(
                        estimate_par.squeeze().detach(), gt_eps.squeeze().detach(), estimate_par=estimate_par, tv_reg=0
                    )
                    plt.title("eps mse:{:.2f}, u mse:{:.2f}".format(mse["mse"], loss_t))
                    print("eps mse:{:.2f}, u mse:{:.2f}".format(mse["mse"], loss_t))
                plt.subplot(133)
                diff = (
                    estimate_par_denorm[:, :, :, 0].squeeze().detach().cpu().numpy()
                    - gt_eps_denorm[:, :, 0].squeeze().detach().cpu().numpy()
                )
                utils.plt_colorbar(diff, vmin=diff.max(), vmax=-1 * (diff.max()), cmap="seismic")
                plt.title("iter:{},diff".format(total_steps))
                if save:
                    plt.tight_layout()
                    plt.savefig("{}/{}.png".format(root_path, total_steps))
                del estimate_par_denorm
                del diff
                del gt_eps_denorm
                # plt.close('all')
                del mse

            if total_steps == pivit_iter and pivit:
                print("start piviting")
                if optimizer == "adam":
                    optim = torch.optim.Adam(
                        [{"params": prior.parameters(), "lr": 1e-6}, {"params": par.parameters(), "lr": lr}]
                    )

    print("use {} frequenies".format(num_f_t))
    plt.figure()
    plt.subplot(131)
    gt_eps_denorm = normalizer.decode(gt_eps.cpu())
    utils.plt_colorbar(gt_eps_denorm[:, :, 0].squeeze().detach().cpu().numpy(), vmin=1, vmax=10)
    plt.title("gt,use{}".format(num_f_t))
    plt.subplot(132)
    estimate_par_denorm = normalizer.decode(estimate_par.cpu())
    utils.plt_colorbar(estimate_par_denorm[:, :, :, 0].squeeze().detach().cpu().numpy(), vmin=1, vmax=10)
    plt.title("iter:{},gt".format(total_steps))

    if gt_eps != None:
        mse = loss_fn(estimate_par.squeeze().detach(), gt_eps.squeeze().detach(), estimate_par=estimate_par, tv_reg=0)
        plt.title("eps mse:{:.2f}, u mse:{:.2f}".format(mse["mse"], loss_t))
        print("eps mse:{:.2f}, u mse:{:.2f}".format(mse["mse"], loss_t))
    plt.subplot(133)
    diff = (
        estimate_par_denorm[:, :, :, 0].squeeze().detach().cpu().numpy()
        - gt_eps_denorm[:, :, 0].squeeze().detach().cpu().numpy()
    )
    utils.plt_colorbar(diff, vmin=diff.max(), vmax=-1 * (diff.max()), cmap="seismic")
    plt.title("iter:{},diff".format(total_steps))
    if save:
        plt.tight_layout()
        plt.savefig("{}/{}.png".format(root_path, total_steps))
    del estimate_par_denorm
    del diff
    del gt_eps_denorm
    # plt.close('all')
    del mse

    return estimate_pars, loss, mses, mses_decoded
