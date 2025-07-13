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

Please make sure to read our documentation and comments in the code files as well as the relevant [Lightly documentation](https://docs.lightly.ai/self-supervised-learning/) to understand how each individual component works. The code is structured in a modular way, allowing you to easily adapt it for your own use cases, if desired.

### Repository structure

- `honeybee_learning/`: Contains all Python code for the project, including the training and evaluation scripts.
  - Each of the Python modules in this package has a docstring at the top of the file that describes its purpose and functionality.
- `batch-*.sh`: Batch scripts that can be used to run the training and evaluation scripts on the RAMSES cluster using `sbatch`. They are provided as examples and can be used as a starting point for your own batch jobs.
  - Of these, we only actually used the `batch-train-*.sh` scripts, as only the training required us to use an NVIDIA H100 GPU due to the memory requirements of the large batch sizes recommended by the SimCLR and VICReg papers.
  - All other results were obtained by running the provided entrypoints via uv in an interactive session on a RAMSES node with an NVIDIA A30 GPU.
- `pyproject.toml`: The project configuration file that defines the dependencies and entrypoints of the project.
  - It is compliant with the [`pyproject.toml` specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#pyproject-toml-spec) and used by uv to manage the dependencies and run the code. For more details, refer to the [Running the code](#running-the-code) section below.
- `uv.lock`: The lock file generated by uv that contains the exact versions of the dependencies used in this project.
- `requirements.txt`: A requirements file that can be used to install the exact versions of the dependencies using any package manager that supports the `requirements.txt` format, such as pip.
  - Nevertheless, for optimal reproducibility, we recommend using uv to install the dependencies, as this is what we used to run the code in this repository. See the [Running the code](#running-the-code) section below for more details.
- Other auxiliary files in the root directory:
  - `README.md`: This file, which provides an overview of the project and how to run the code.
  - `flake.nix`: A [Nix flake](https://nixos.wiki/wiki/Flakes) that can be used to create a development environment with Python 3.13 and uv installed. This is useful if you are familiar with the [Nix package manager](https://nixos.org/) and want to use it to manage your development environment.
  - `flake.lock`: The lock file for the Nix flake, which contains the exact revision of [Nixpkgs](https://github.com/NixOS/nixpkgs) to source the Python and uv derivations from.
  - `.gitignore`: A file that specifies which files and directories should be ignored by Git when committing changes to the repository.
  - `.envrc`: Used by [direnv](https://direnv.net/) to automatically set up the development environment when entering the project directory. It is not required to run the code, but it can be useful if you want to use direnv to manage your development environment.

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

The entrypoints below are designed to be run in the order they are listed here. However, if you already have the necessary files in place, you can also run individual entrypoints independently.

#### `preprocessing`

The `preprocessing` entrypoint is used to extract invididual frames from the video footage of the honeybee tracking dataset and use the trajectories provided in the dataset to crop the individual honeybee images from the extracted frames. Refer to the `preprocessing` module for more details on how the preprocessing works. The entrypoint can be run using the following command:

```shell
uv run preprocessing
```

#### `train`

The `train` entrypoint is the entrypoint for training the models. It is in fact a small CLI that allows you to select the model architecture to use (SimCLR or VICReg) and whether to log the training run to Weights & Biases or not. For details on its usage, run `uv run train --help`. For example, to train a SimCLR model with Weights & Biases logging enabled, you can run the following command:

```shell
uv run train --model simclr --wandb
```

The hyperparameters used for training the models can be configured in the `honeybee_learning/simclr.py` and `honeybee_learning/vicreg.py` files, respectively.

#### `precalculate-representations`

The `precalculate-representations` entrypoint is used to precalculate the representations of the honeybee images using a trained model. This is useful to speed up the evaluation of the models, as it allows you to calculate the representations once and reuse them for different evaluation tasks. Like `train`, this entrypoint is a small CLI. It requires you to select which model checkpoint to use for the representation calculation and the model architecture the checkpoint was generated for (SimCLR or VICReg). For details on its usage, run `uv run precalculate-representations --help`. For example, to precalculate the representations of the honeybee images using a SimCLR model, you can run the following command:

```shell
uv run precalculate-representations --model simclr <path_to_checkpoint>
```

#### `test-linear`

The `test-linear` entrypoint is used to evaluate the performance of a set of linear predictors trained in supervised fashion on the representations calculated by a self-supervised representation learning model with the labels of the honeybee attributes provided in the dataset as targets. This is a common evaluation method for self-supervised representation learning models, as it allows you to assess how well the representations learned by the model can be used for downstream tasks. This entrypoint is also a small CLI that requires you to specify the path to the serialized NumPy array file containing the representations to use for evaluation, as well as whether to log the training and testing metrics to Weights & Biases or not. For details on its usage, run `uv run test-linear --help`. For example, to run the evaluation on a specific set of representations and log the metrics to Weights & Biases, you can run the following command:

```shell
uv run test-linear --wandb <path_to_representations>
```

#### `visualize`

The `visualize` entrypoint is used to visualize the representations calculated by a self-supervised representation learning model. It creates a set of figures that show the representations in a 3D space, allowing you to visually inspect the learned representations and how well they separate different honeybee attributes. This entrypoint is also a small CLI that requires you to specify the path to the serialized NumPy array file containing the representations to visualize. For details on its usage, run `uv run visualize --help`. For example, to visualize a specific set of representations, you can run the following command:

```shell
uv run visualize <path_to_representations>
```
