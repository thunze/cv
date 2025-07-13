#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH -G h100:1
#SBATCH --mem=512gb
#SBATCH --time=1:00:00

set -e
set -u
set -o pipefail
set -x

pip install uv==0.7.13
uv run visualize /scratch/cv-course2025/group7/representations/vicreg_20250708_114240_epoch_20_representations_2048_first_relu.npy
