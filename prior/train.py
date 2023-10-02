# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import cProfile
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

import utils


def train(
    model,
    train_dataloader,
    epochs,
    lr,
    steps_til_summary,
    epochs_til_checkpoint,
    model_dir,
    loss_fn,
    summary_fn,
    prefix_model_dir="",
    val_dataloader=None,
    double_precision=False,
    loss_schedules=None,
    params=None,
    lr_latent=None,
    wandb=None,
):

    if lr_latent == None:
        optim = torch.optim.Adam(lr=lr, params=model.parameters())
    else:
        optim = torch.optim.Adam(
            [
                {"params": model.decoder.parameters(), "lr": lr},
                {"params": model.lat_vecs.parameters(), "lr": lr_latent},
            ]
        )

    if os.path.exists(model_dir):
        pass
    else:
        os.makedirs(model_dir)

    model_dir_postfixed = os.path.join(model_dir, prefix_model_dir)

    summaries_dir = os.path.join(model_dir_postfixed, "summaries")
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir_postfixed, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(
                    {"model_state_dict": model.state_dict(), "optimizer_state_dict": optim.state_dict()},
                    os.path.join(checkpoints_dir, "model_epoch_%04d.pth" % epoch),
                )
                np.savetxt(os.path.join(checkpoints_dir, "train_losses_epoch_%04d.txt" % epoch), np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = model_input.cuda()

                gt = gt.cuda()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.0
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    if wandb != None:
                        wandb.log({loss_name: single_loss}, step=total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)
                if wandb != None:
                    wandb.log({"total_train_loss": train_loss}, step=total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(), os.path.join(checkpoints_dir, "model_current.pth"))
                    summary_fn(
                        model,
                        model_input,
                        gt,
                        model_output,
                        writer,
                        total_steps,
                        "train",
                    )

                optim.zero_grad()
                train_loss.backward()

                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    summary_fn(
                        model,
                        model_input,
                        gt,
                        model_output,
                        writer,
                        total_steps,
                        "val",
                    )
                    tqdm.write(
                        "Epoch %d, Total loss %0.6f, iteration time %0.6f"
                        % (epoch, train_loss, time.time() - start_time)
                    )

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                gt = gt.permute(0, 3, 1, 2)
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, "model_final.pth"))
        np.savetxt(os.path.join(checkpoints_dir, "train_losses_final.txt"), np.array(train_losses))


class LinearDecaySchedule:
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.0)


def dict2cuda(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cuda()})
        else:
            tmp.update({key: value})
    return tmp


def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        else:
            tmp.update({key: value})
    return tmp
