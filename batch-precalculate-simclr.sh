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
uv run precalculate-representations --model simclr /scratch/cv-course2025/group7/checkpoints/simclr_20250704_213650_epoch_90.pth
