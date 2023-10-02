<!--
Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Learned Born Operator for Reflection Tomographic Imaging

This repository includes source code for training and using the Born Fourier Neural Operator (BornFNO) model proposed in our ICASSP 2023 paper,
**Deep Born Operator Learning for Reflection Tomographic Imaging**
by Qingqing Zhao, Yanting Ma, Petros Boufounos, Saleh Nabi, Hassan Mansour.

[Please click here to read the paper.](https://www.merl.com/publications/TR2023-029)

If you use any part of this code for your work, we ask that you include the following citation:

    @InProceedings{Zhao_2023ICASSP,
      author =	 {Qingqing Zhao and Yanting Ma and Petros Boufounos and Saleh Nabi and Hassan Mansour},
      title =	 {Deep Born Operator Learning for Reflection Tomographic Imaging},
      booktitle =	 {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year =	 2023,
      month =	 June
    }


## Table of contents

1. [Environment Setup](#environment-setup)
2. [Download the Dataset and Pretrained Models](#download-the-dataset-and-pretrained-models)
3. [Training and Evaluating the Forward Models](#training-and-evaluating-the-forward-models)
4. [Training the Autoencoder Prior Network](#training-the-autoencoder-prior-network)
5. [Solving the Inverse Problem](#solving-the-inverse-problem)


## Environment Setup

To setup a conda environment use these commands

```bash
conda env create -f environment.yml
conda activate deepGPR
```

## Download the Dataset and Pretrained Models
Dataset and the pretrained model can be download [here](https://zenodo.org/record/8145084). Unzip the `data.zip` and place `v1` it in the `data` folder. In this dataset, we consider two layer structures with two cylinders of various radius embedded completely in the second layer. More setup detail could be found in the paper, section 4.1.<br>
The structure of h5 file:


    data_frequency.h5
        ├── 0
        │   └── eps
            └── f_field
        ├── 1
            └── eps
            └── f_field
        ├── ...

Unzip the `model_zoo.zip` and place it in the `./` folder.

## Training and Evaluating the Forward Models
To start multiple training in parallel, run following commends. The default setting includes FNO/BornFNO with 1/5 layers.
 ```bash
cd forward_model
python run_fno.py
```
To train one BornFNO model only, use following commands. Modify `layer_num`, `type` if necessary,
```bash
cd forward_model
python train_fno.py --type="born" --layer_num=5 --modes=12 --width=128 --lr=1e-3 --batch_size=64 --dataset=v1 --add_noise=0 --step_size=700
```
The log and the trained model will be saved in `model_zoo/forward_model/`.
To evaluate the trained model, run [forward_model/eval.ipynb](https://github.com/merl-oss-private/DeepBornFNO_private/blob/main/forward_model/eval.ipynb).
## Training the Autoencoder Prior Network
```bash
cd prior
python experiment_prior.py --sig=0.001 --latent_size=64 --gpu=6
```
The log and the trained model will be saved in `model_zoo/priors/`.

## Solving the Inverse Problem
```bash
cd inverse_problem
python run_results_no_prior.py # solve inverse problem without learned autoencoder prior
python run_results_with_prior.py # solve inverse problem with learned autoencoder prior
```
Run [inverse_problem/demo.ipynb](https://github.com/merl-oss-private/DeepBornFNO_private/blob/main/inverse_problem/demo.ipynb) for a quick demo and visualization.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```

`forward_model/Adam.py` was adapted from an earlier release of neuraloperator https://github.com/neuraloperator/neuraloperator and `forward_model/model.py` contains functions that were adapted and modified from an earlier release of neuraloperator https://github.com/neuraloperator/neuraloperator (`MIT` license as found in [LICENSES/MIT.txt](LICENSES/MIT.txt)):

```
Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2023 NeuralOperator developers

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: MIT
```

`prior/encoder_model.py` was adapted and modified from https://github.com/AntixK/PyTorch-VAE/ (`Apache-2.0` license as found in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md))

```
Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2020 Anand Krishnamoorthy Subramanian

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: Apache-2.0
```
