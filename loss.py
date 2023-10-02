# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch


# loss function with rel/abs Lp loss
class LpLoss(object):
    """
    @author: Zongyi Li
    This function is the taken from Zongyi Li's code for work [paper](https://arxiv.org/pdf/2010.08895.pdf).
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(((x - y)).reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm((y).reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def encoder(loss_mode, sig, prediction, target):
    if loss_mode == "l2":
        latent = prediction["latent"]
        output = prediction["output"]
        return {"mse": ((output - target) ** 2).mean(), "latent": sig * ((latent**2).mean())}


def mse(prediction, target, mask=torch.tensor(1), estimate_par=None, tv_reg=0):
    if tv_reg == 0:
        tvloss = 0
    else:
        tvloss = total_variation_loss(estimate_par, weight=tv_reg)
    if len(mask.shape) > 1:
        return {"mse": ((prediction * mask - target) ** 2).sum() / (mask.sum()) / prediction.shape[0], "tv": tvloss}
    return {"mse": ((prediction * mask - target) ** 2).mean(), "tv": tvloss}


def l1(prediction, target, mask=torch.tensor(1), estimate_par=None, tv_reg=0):
    tvloss = total_variation_loss(estimate_par, weight=tv_reg)
    if len(mask.shape) > 1:
        return {"mse": ((prediction * mask - target).abs()).sum() / (mask.sum()) / prediction.shape[0], "tv": tvloss}
    return {"mse": ((prediction * mask - target).abs()).mean(), "tv": tvloss}


def mse_unet(prediction, target):
    return {"mse": ((prediction - target) ** 2).mean()}


def mse_fno(sum_loss, pde_factor, iv_factor, pred, gt, prefix="gt"):
    if not sum_loss:
        mse = ((pred["pred"].squeeze() - gt[prefix].reshape(pred["pred"].squeeze().shape)) ** 2).mean()
        return {"mse": mse}
    else:
        if "pred" in pred.keys():
            mse_iv = ((pred["pred"].squeeze() - gt["gt"].reshape(pred["pred"].squeeze().shape)) ** 2).mean()
        else:
            mse_iv = 0
        mse_pde = ((pred["out_sim"].squeeze() - gt["y"].reshape(pred["out_sim"].squeeze().shape)) ** 2).mean()
        return {"mse_iv": iv_factor * mse_iv, "mse_pde": mse_pde * pde_factor}


def total_variation_loss(img, weight, type="isotropic"):
    if type == "isotropic":
        img = img.permute(0, 3, 1, 2)
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:, :, 1:, 1:] - img[:, :, :-1, 1:], 2)
        tv_w = torch.pow(img[:, :, 1:, 1:] - img[:, :, 1:, :-1], 2)
        return weight * ((torch.sqrt(tv_h + tv_w)).sum()) / (bs_img * c_img * h_img * w_img)
