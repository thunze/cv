#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH -G h100:1
#SBATCH --mem=512gb
#SBATCH --time=3-0:00:00

set -e
set -u
set -o pipefail
set -x

pip install uv==0.7.13
uv run train --model vicreg --wandb
