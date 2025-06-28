"""Configuration for the honeybee representation learning project."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "TOTAL_NUMBER_OF_BEES",
    "DEVICE",
    "DATALOADER_NUM_WORKERS",
    "CHECKPOINTS_PATH",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
]


# Data
TOTAL_NUMBER_OF_BEES = 874  # Total number of unique bees in the honeybee dataset


# Training and validation
DEVICE = "cuda"  # Device to train and run the model on, typically a GPU
DATALOADER_NUM_WORKERS = 8  # Number of processes used by `DataLoader` instances


# Paths
CHECKPOINTS_PATH = Path("./checkpoints")  # Path to save checkpoints to


# Weights & Biases (wandb) configuration
WANDB_ENTITY = "thunze"  # Team name
WANDB_PROJECT = "honeybee-learning"  # Project name
