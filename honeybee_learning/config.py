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
    "CROPS_PATH",
    "TRAIN_RATIO",
    "VALIDATION_RATIO",
    "SHUFFLE",
    "SEED",
    "MAX_FRAME_DIFFERENCE"
]


# Data
TOTAL_NUMBER_OF_BEES = 874  # Total number of unique bees in the honeybee dataset
MAX_FRAME_DIFFERENCE = 5 # Number of frames that will still count a bee as temporally adjacent

# Training and validation
DEVICE = "cuda"  # Device to train and run the model on, typically a GPU
DATALOADER_NUM_WORKERS = 8  # Number of processes used by `DataLoader` instances
TRAIN_RATIO = 0.4   # Amount of the data to use for training
VALIDATION_RATIO = 0.4 # Amount of the data to use for validation

# Paths
CHECKPOINTS_PATH = Path("./checkpoints")  # Path to save checkpoints to
CROPS_PATH = Path("/scratch/cv-course2025/group7/crops128/")

# Weights & Biases (wandb) configuration
WANDB_ENTITY = "thunze"  # Team name
WANDB_PROJECT = "honeybee-learning"  # Project name

# Randomization
SHUFFLE = True  # Whether to use shuffling in dataset creation or not
SEED = 42   # Seed to use for shuffling operations