# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import subprocess
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

processes = set()
max_processes = 8


vals = [
    (
        lr,
        in_f,
        reg_coeff,
        sweep_par,
        num_iter,
        mask,
        drop_reg_high,
        prior,
        pivit,
        optimizer,
        type,
        note,
        filter,
        gamma,
        latent,
        noise_src,
        load_weight,
        fno_noise,
    )
    for pivit in [[1, 300]]
    for sweep_par in ["fno"]
    for num_iter in [1200]
    for mask in ["line"]
    for lr in [0.1]
    for reg_coeff in [0.01]
    for in_f in [1]
    for drop_reg_high in [0]
    for prior in ["None"]
    for optimizer in ["adam"]
    for type in [
        ["born", 10, 128],
    ]
    for note in ["log"]
    for filter in [1]
    for gamma in [0.5]
    for noise_src in [0]
    for latent in [64]
    for load_weight in [1]
    for fno_noise in [0]
]

gpu = [0, 1, 2, 3]
gpu_idx = 0
process_idx = 0
for idx, val in enumerate(vals):
    lr = val[0]
    in_f = val[1]
    reg_coeff = val[2]
    sweep_par = val[3]
    num_iter = val[4]
    mask = val[5]
    drop_reg_high = val[6]
    prior = val[7]
    pivit = val[8][0]
    pivit_iter = val[8][1]
    optimizer = val[9]
    type = val[10][0]
    fno_par1 = val[10][1]
    fno_par2 = val[10][2]
    note = val[11]
    filter = val[12]
    gamma = val[13]
    latent = val[14]
    noise_src = val[15]
    load_weight = val[16]
    fno_noise = val[17]
    sweep_par = "{}_{}".format(mask, sweep_par)
    if prior == "None":
        pivit = 0

    processes.add(
        subprocess.Popen(
            [
                "python",
                "iter_solve_inverse.py",
                "--lr",
                f"{lr}",
                "--gpu",
                f"{gpu[gpu_idx]}",
                "--in_f",
                f"{in_f}",
                "--reg_coeff",
                f"{reg_coeff}",
                "--sweep_par",
                f"{sweep_par}",
                "--mask",
                f"{mask}",
                "--num_iter",
                f"{num_iter}",
                "--drop_reg_high",
                f"{drop_reg_high}",
                "--prior",
                f"{prior}",
                "--pivit",
                f"{pivit}",
                "--pivit_iter",
                f"{pivit_iter}",
                "--optimizer",
                f"{optimizer}",
                "--filter",
                f"{filter}",
                "--gamma",
                f"{gamma}",
                "--latent",
                f"{latent}",
                "--fno_noise",
                f"{fno_noise}",
                "--noise_src",
                f"{noise_src}",
                "--load_weight",
                f"{load_weight}",
                "--type",
                f"{type}",
                "--fno_par",
                f"{fno_par1}",
                f"{fno_par2}",
                "--note",
                f"{note}",
            ]
        )
    )

    gpu_idx += 1
    while gpu_idx >= len(gpu):
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
        if len(processes) == 0:
            gpu_idx = 0


# Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()
