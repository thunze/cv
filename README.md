# Representation Learning of Honeybees

This repository contains the code for the project “Representation Learning of Honeybees”, which is about applying unsupervised representation learning techniques to cropped images of honeybees taken from the video footage provided with the [Honeybee Tracking I Dataset](https://groups.oist.jp/bptu/honeybee-tracking-dataset#pid) of the [Honeybee Segmentation and Tracking Datasets](https://groups.oist.jp/bptu/honeybee-tracking-dataset) by the Okinawa Institute of Science and Technology (OIST).

In particular, we apply both **SimCLR**, a contrastive learning method, and **VICReg**, an information maximization method, to the honeybee images. The goal is to learn useful representations of the honeybees that can be used for downstream tasks such as classification and regression of the honeybee attributes provided as labels in the dataset, as well as for visualization of the learned representations.

Please refer to the project report for more details on the methods used for training and evaluating the models, as well as the results obtained.

You can find logs and live metrics of different training runs on [Weights & Biases](https://wandb.ai/thunze/honeybee-learning).

## About the dataset

The dataset used in this project is the [Honeybee Tracking I Dataset](https://groups.oist.jp/bptu/honeybee-tracking-dataset#pid) of the [Honeybee Segmentation and Tracking Datasets](https://groups.oist.jp/bptu/honeybee-tracking-dataset) by the Okinawa Institute of Science and Technology (OIST). Both provided recordings were used for this project. For each recording, the dataset provides:

- `video`: Video footage of a honeybee comb, which was recorded for 5 minutes at 10 frames per second (fps).
- `detections`: A set of detections of individual honeybees in the video footage. We did not use these annotations in our project.
- `trajectories`: A set of trajectories of individual honeybees in the video footage.

The trajectories are provided as a `.tgz` file, which contains a list of `<bee_id>.txt` files, where each file contains the trajectories of the honeybee with the unique ID `bee_id`. Within each file, each line contains one trajectory point in the format `<frame_nb>, <position_x>, <position_y>, <object_class>, <orientation_angle>`. The meaning of the individual fields is as follows:

- `frame_nb`: The frame number of the trajectory point. Note that some bees may not be present in all frames, so the frame numbers may not be consecutive.
- `position_x`: The **y**-coordinate of the trajectory point in pixels.
- `position_y`: The **x**-coordinate of the trajectory point in pixels.
- `object_class`: The class of the object.
    - **0** means the bee is fully visible.
    - **1** means the abdomen of the bee is partially hidden inside a cell of the honeycomb at that particular frame.
- `orientation_angle`: The orientation angle of the bee in degrees.
  - **0** degrees means the bee is facing upwards.
  - **90** degrees means the bee is facing to the right.
  - **180** degrees means the bee is facing downwards.
  - **270** degrees means the bee is facing to the left.

> [!CAUTION]
> The above _column names_ are the official names used by the dataset creators. However, we found that the `position_x` and `position_y` values are actually swapped in the dataset. In fact, the second column of each line contains the **y**-coordinate, while the third column contains the **x**-coordinate. This is not an error on our end, but rather a mislabeling in the dataset, which is taken into account in the code and documented in the above description of the fields.

## About the code

The code is written in [Python](https://www.python.org/) and uses [PyTorch](https://pytorch.org/) as the main framework for training the models.

None of the code you can find in this repository is forked or directly copied from other repositories. Instead, the code makes use of the [Lightly](https://github.com/lightly-ai/lightly) library, which provides PyTorch implementations of various unsupervised representation learning methods, including SimCLR and VICReg, plus utilities to facilitate training these models (e.g., an implementation of the LARS optimizer as recommended by the SimCLR and VICReg papers). The training code is _loosely_ based on the examples provided in the [Lightly documentation](https://docs.lightly.ai/self-supervised-learning/) and the [benchmark implementations](https://github.com/lightly-ai/lightly/tree/master/benchmarks/imagenet/resnet50) of the models (this is their take at reproducing the results of the corresponding papers), but has been heavily adapted to fit the specific use case of this project.

**TODO:** A list of files and what they contain.

Please make sure to read our documentation and comments in the code files as well as the relevant [Lightly documentation](https://docs.lightly.ai/self-supervised-learning/) to understand how each individual component works. The code is structured in a modular way, allowing you to easily adapt it for your own use cases, if desired.

## Running the code

Running the code requires **Python 3.13** and a Python environment with the necessary dependencies installed.

This repository uses [uv](https://docs.astral.sh/uv/) as the package manager. The last tested version of uv in the context of this project is `0.7.13`. You can install `uv` as detailed in the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). If you are familiar with [Nix](https://nixos.org/), you may also simply run `nix develop` in the root directory of the repository to enter a development shell with Python 3.13 and uv installed.

The dependencies of this project are defined in the `pyproject.toml` file in the root directory of the repository. The dependency specifications in this file are compliant with [PEP 621](https://peps.python.org/pep-0621/), which is supported by uv. While the `pyproject.toml` file specifies dependency constraints, we recommend installing the dependencies of this project using uv to make use of the `uv.lock` file, which contains the exact versions of the dependencies that were used to run the code in this repository.

Alternatively, we also provide a `requirements.txt` file that can be used to install the dependencies using any package manager that supports the `requirements.txt` format, such as pip. This file was generated by uv using the `uv export` command and, just like the `uv.lock` file, contains the exact versions of the dependencies that were used to run the code in this repository. Nevertheless, for optimal reproducibility, we recommend using uv to install the dependencies. Please don't hesitate to reach out if you have any questions or issues with the installation process.

To run the code, we provide several entrypoints as **uv scripts** in the `pyproject.toml` file. These scripts can be run using the `uv run <entrypoint>` command, which will automatically set up the environment and run the script with the correct dependencies. Please refer to the [documentation of the individual entrypoints](#entrypoints) below for more details on how to use them.

If you instead want to create a new virtual environment and install the necessary dependencies using uv, for example to explore the code with proper autocompletion and type checking, you can do so by running the following commands in the root directory of the repository:

```shell
uv venv .venv
source .venv/bin/activate
uv sync
```

### Entrypoints

For all of the below entrypoints, make sure you have prepared the working directories the individual entrypoints depend on to conform to the expected structure. The individual paths to these directories can be configured in `honeybee_learning/config.py`. For reference, as we had to create these directories ourselves to successfully run preprocessing, training and evaluation, you can find the working directories we used in `/scratch/cv-course2025/group7` on RAMSES. Refer to `honeybee_learning/config.py` for the exact paths we used.

**TODO:** Rename these?

#### `preprocessing`

The `preprocessing` entrypoint is used to extract invididual frames from the video footage of the honeybee tracking dataset and use the trajectories provided in the dataset to crop the individual honeybee images from the extracted frames. Refer to the `preprocessing` module for more details on how the preprocessing works. The entrypoint can be run using the following command:

```shell
uv run preprocessing
```

#### `honeybee-learning`

The `honeybee-learning` entrypoint is the entrypoint for training the models. It is in fact a small CLI that allows you to select the model architecture to use (SimCLR or VICReg) and whether to log the training run to Weights & Biases or not. For details on its usage, run `uv run honeybee-learning --help`. For example, to train a SimCLR model with Weights & Biases logging enabled, you can run the following command:

```shell
uv run honeybee-learning --model simclr --wandb
```

The hyperparameters used for training the models can be configured in the `honeybee_learning/simclr.py` and `honeybee_learning/vicreg.py` files, respectively.
