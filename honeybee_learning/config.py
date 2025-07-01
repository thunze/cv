"""Configuration for the honeybee representation learning project."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "CHECKPOINTS_PATH",
    "CROPS_PATH",
    "TOTAL_NUMBER_OF_BEES",
    "MAX_FRAME_DIFFERENCE",
    "DATASET_CREATE_SHUFFLE",
    "DATASET_CREATE_SHUFFLE_SEED",
    "TRAIN_RATIO",
    "VALIDATION_RATIO",
    "DEVICE",
    "DATALOADER_NUM_WORKERS",
    "CHECKPOINT_EVERY_N_EPOCHS",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
]


# Paths
CHECKPOINTS_PATH = Path("/scratch/cv-course2025/group7/checkpoints")
CROPS_PATH = Path("/scratch/cv-course2025/group7/crops128/")

# Data
TOTAL_NUMBER_OF_BEES = 876  # Total number of unique bees in the honeybee dataset
MAX_FRAME_DIFFERENCE = 5  # Max. number of frames that count bee as temporally adjacent
DATASET_CREATE_SHUFFLE = True  # Whether to use shuffling in dataset creation or not
DATASET_CREATE_SHUFFLE_SEED = 42  # Seed to use when shuffling in dataset creation
TRAIN_RATIO = 0.4  # Amount of the data to use for training
VALIDATION_RATIO = 0.4  # Amount of the data to use for validation

# Training and validation
DEVICE = "cuda"  # Device to train and run the model on, typically a GPU
DATALOADER_NUM_WORKERS = 8  # Number of processes used by `DataLoader` instances
CHECKPOINT_EVERY_N_EPOCHS = 10  # Save model checkpoint every N epochs

# Weights & Biases (wandb) configuration
WANDB_ENTITY = "thunze"  # Team name
WANDB_PROJECT = "honeybee-learning"  # Project name
