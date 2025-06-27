"""Configuration for the honeybee representation learning project."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "DEVICE",
    "DATALOADER_NUM_WORKERS",
    "CHECKPOINTS_PATH",
]


# Training and validation
DEVICE = "cuda"  # Device to train and run the model on, typically a GPU
DATALOADER_NUM_WORKERS = 8  # Number of processes used by `DataLoader` instances


# Paths
CHECKPOINTS_PATH = Path("./checkpoints")  # Path to save checkpoints to


# Weights & Biases (wandb) configuration
WANDB_ENTITY = "thunze"  # Team name
WANDB_PROJECT = "honeybee-learning"  # Project name
