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
    (type, modes, width, lr, batch_size, dataset, add_noise, step_size)
    for type in [["born", 5], ["born", 1]]
    for modes in [12]
    for width in [128]
    for lr in [1e-3]
    for batch_size in [64]
    for dataset in ["v1"]
    for add_noise in [0]
    for step_size in [700]
]


total_process = 2
process_idx = 0
gpu = [0, 1]  # [2,3]
gpu_idx = 0
for idx, val in enumerate(vals):
    type = val[0][0]
    layer_num = val[0][1]
    modes = val[1]
    width = val[2]
    lr = val[3]
    batch_size = val[4]
    dataset = val[5]
    add_noise = val[6]
    step_size = val[7]

    processes.add(
        subprocess.Popen(
            [
                "python",
                "train_fno.py",
                "--type",
                f"{type}",
                "--layer_num",
                f"{layer_num}",
                "--gpu",
                f"{gpu[idx]}",
                "--modes",
                f"{modes}",
                "--width",
                f"{width}",
                "--lr",
                f"{lr}",
                "--batch_size",
                f"{batch_size}",
                "--dataset",
                f"{dataset}",
                "--add_noise",
                f"{add_noise}",
                "--step_size",
                f"{step_size}",
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
